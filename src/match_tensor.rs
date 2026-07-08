#[cfg(feature="match-tensor")]
impl CellPattern{
	pub fn default_transform_cost <S:AsRef<str>>(&self,s:S)->f32{self.diff_lev_transform_cost(1.0,1.0,1.0,5.0,s)}
	/// builtin transformation cost function that uses levenstein*levscale for strings, xor*boolscale for bool, difference*diffscale for numbers, and parsescale when the string is not parsable
	pub fn diff_lev_transform_cost<S:AsRef<str>>(&self,boolscale:f32,diffscale:f32,levscale:f32,parsescale:f32,s:S)->f32{
		static METRIC:OnceLock<Levenshtein>=OnceLock::new();
		let s=s.as_ref();

		match self{
			CellPattern::Anything=>0.0,
			CellPattern::Bool(e)=>if let Ok(x)=s.parse::<bool>(){((e^x) as usize) as f32*boolscale}else{parsescale},
			CellPattern::Boolean=>if let Ok(_x)=s.parse::<bool>(){0.0}else{parsescale},
			CellPattern::Nothing=>if s.len()==0{0.0}else{parsescale},
			CellPattern::Number(e)=>if let Ok(x)=s.parse::<f64>(){((e-x).abs()*diffscale as f64) as f32}else{parsescale},
			CellPattern::Numeric=>if let Ok(_x)=s.parse::<f64>(){0.0}else{parsescale},
			CellPattern::Static(z)=>METRIC.get_or_init(Levenshtein::new).distance(s,z) as f32*levscale,
			CellPattern::Text(z)=>METRIC.get_or_init(Levenshtein::new).distance(s,z) as f32*levscale
		}
	}
}

#[cfg(test)]
mod tests{
	#[test]
	fn fill_holes_00(){
		let mut data:Tensor<f32>=vec![
			0.5,0.0,0.5,0.4,0.1,
			0.0,0.1,0.0,0.4,0.2,
			0.0,0.0,0.4,0.0,0.3,
		].into();
		let filled=vec![
			0.5,0.0,0.5,0.4,0.1,
			0.5,0.1,0.5,0.4,0.2,
			0.5,0.1,0.4,0.4,0.3,
		];

		data.reshape([3,5]);
		fill_holes(&mut data,0,|&x|x,|&x|x==0.0);

		assert_eq!(data.into_flat_vec(None),filled);
	}

	use super::*;
}

#[derive(Clone,Debug,Default,PartialEq)]
/// cell type usable for pattern matching
pub enum CellPattern{
	#[default]
	/// any value
	Anything,
	/// parable bool equal to
	Bool(bool),
	/// parsable bool
	Boolean,
	/// specifically an empty cell
	Nothing,
	/// parsable number equal to
	Number(f64),
	/// parsable number
	Numeric,
	/// text equal to
	Static(&'static str),
	/// text equal to
	Text(String)
}

#[track_caller]
/// find the maximum position by some measure. panics if the tensor is empty or fails to index.
pub fn arg_max_by_key<E,K:PartialOrd>(data:impl AsRef<View<E>>,mut f:impl FnMut(&E)->K)->Position{
	let mut best=None;
	let data=data.as_ref();
	let mut position=Position::new(data.rank());

	for ix in data.indices(){
		let f=f(&data[&ix]);
		if {
			if let Some(x)=&best{f>*x}else{true}
		}{
			best=Some(f);
			position.clone_from(&ix);
		}
	}
	assert!(best.is_some());
	position
}
#[track_caller]
/// Perform a binary search of a lexicographically ordered multidimensional index space using mixed-radix arithmetic. For correct behavior, dims should be <= isize::MAX, and the components from which the underlying ordering is derived should have an order that would make them sorted if they were flattened into a vec
/// check(ix) can be read as comparing a candidate component at ix to a query as if by candidate.partial_cmp(query)
/// Failed comparisons (check returns None) are interpreted with the ordering of the nearest successful comparison at a smaller index. If no such comparison exists, it's interpreted as less.
/// Ok(ix) indicates a successful search where check(ix)==Some(Equal).
/// Err(ix) means no equal candidate was found, where ix is the insertion position in lexicographic order, even if actually inserting like that in a tensor would be inconvenient.
/// When the insertion position is at the end and the returned index has to be one past the last valid index, ix[0] is equal to dims[0] and ix[1..] are all 0.
/// For a scalar, dims==[], and the only index ix==[]. The result in this case is Ok([]) if check([])==Some(Equal), and Err([]) otherwise
/// For an empty tensor, dims contains 0, and no valid indices exist. The result in this case is an Err containing all 0s.
pub fn binary_search_by<F:FnMut(Position)->Option<Ordering>>(mut check:F,dims:&[usize])->Result<Position,Position>{
	let rank=dims.len();						// get rank and allocate mid position
	let mut mid=Position::new(rank);
												// special cases: empty tensor -> Err(0), scalar -> ix=[], err if not equal
	if dims.contains(&0){return Err(mid)}
	if rank==0{
		return if check(mid.clone())==Some(Ordering::Equal){Ok(mid)}else{Err(mid)}
	}
												// allocate start (inclusive) and stop (exclusive) positions, and a temp position in case we need to remember the original midpoint to fall back after probing
	let mut start=Position::new(rank);
	let mut stop =Position::new(rank);
	let mut temp =Position::new(rank);
												// give stop 1 past the last valid index along axis 0. the other indices are still 0 as set by Position::new
	stop[0]=dims[0] as isize;
	// helper function to find a successful comparison by linear probing backwards from mid (exclusive) to start (inclusive). If mid is modified, its final state is an exclusive upper bound on the query's index. If no successful comparison can be found, the result is less and mid is unchanged
	let linear_probe=|check:&mut F,mid:&mut Position,start:&Position,temp:&mut Position|{
		temp.copy_from_slice(mid);
		loop{
			if start==temp{break Ordering::Less}
			temp.rewind(dims);

			if let Some(r)=check(temp.clone()){
				if r==Ordering::Greater{mid.copy_from_slice(&temp)}
				break r;
			}
		}
	};
												// binary search in a mixed radix with digit counts given by dims
	loop{
		let mut remainder=0;					// long divide stop-start by 2
		for n in 0..rank{
			let dif=(stop[n]-start[n])  as usize;
			let rem=(dims[n]*remainder) as usize;
												// the first digit result is the first digit input divided by the divisor. add the appropriately rescaled remainder to the next digit so its result can be computed the same way.
			mid[n]   =((dif+rem)/2) as isize;
			remainder= (dif+rem)%2;
		}
												// find a valid comparison
		match check(mid.clone()).unwrap_or_else(||linear_probe(&mut check,&mut mid,&start,&mut temp)){
			Ordering::Equal=>return Ok(mid),	// return exact midpoint if equal
			Ordering::Less =>{					// mid is less, so binary search starting from mid+1
				mid.advance(dims);
				start.copy_from_slice(&mid);
			},
			Ordering::Greater=>{				// mid is greater, so binary search stopping excluded at mid
				stop .copy_from_slice(&mid);
			}
		}
		if start==stop{break}
	}											// component not found
	Err(mid)
}
#[track_caller]
/// compare regions of a tensor to a template
pub fn compare_subtensors<'a,E:'a,X:'a,Y:'a>(data:impl 'a+AsRef<View<E>>,mut f:impl 'a+FnMut(&View<E>,&View<X>,Position)->Y,template:impl 'a+AsRef<View<X>>)->impl 'a+Iterator<Item=Y>{
	let mut layout=data.as_ref().get_layout();
	let rd=template.as_ref().dims();
	let r =data.as_ref().rank();

	assert_eq!(r,rd.len());

	let ld=layout.dims_mut();
	for n in 0..r{ld[n]=ld[n].saturating_sub(rd[n].saturating_sub(1))}

	let mut ranges=vec![0..0;ld.len()];
	GridIter::from_shared_layout(layout.clone()).map(move|ix|{
		let rd=template.as_ref().dims();
		for n in 0..r{ranges[n]=ix[n]..ix[n]+rd[n] as isize}

		f(&data.as_ref().view_ref().slice(&ranges),template.as_ref(),ix)
	})
}
#[track_caller]
/// fills holes in the data using data from the left. holes with no left non holes will remain
pub fn fill_holes<E,F:FnMut(&E)->E,G:FnMut(&E)->bool>(data:&mut View<E>,dim:isize,mut fill_hole:F,mut is_hole:G){
	let mut data=data.view_mut().swap_dims(dim,-1);
	let l=data.rank();
	let mut left=Position::new(0);

	for ix in data.indices(){
		if is_hole(&data[&ix]){
			if left.len()>0&&left[..l-1]==ix[..l-1]{data[&ix]=fill_hole(&data[&left])}
		}
		left.clone_from(&ix);
	}
}
#[track_caller]
/// find the closest match of a string. panics if the tensor is empty or fails to index.
pub fn fuzzy_str_pos<E:AsRef<str>>(data:impl AsRef<View<E>>,query:&str)->Position{
	static METRIC:OnceLock<Levenshtein>=OnceLock::new();
	arg_max_by_key(data,|e|usize::MAX-METRIC.get_or_init(Levenshtein::new).distance(e.as_ref(),query))
}
/// fuzzy finds a table based on the cell pattern
pub fn grab_table<E:Clone,X>(data:impl AsRef<View<E>>,mut f:impl FnMut(&E,&X)->f32,pattern:impl AsRef<View<X>>)->(Tensor<E>,f32){
	let (data,mut pattern)=(data.as_ref().view_ref(),pattern.as_ref().view_ref());

	assert!(data.rank()>=pattern.rank());
	while data.rank()>pattern.rank(){pattern=pattern.unsqueeze_dim(0)}

	//let patterndims=pattern.dims();
	let mut ranges:Vec<Range<usize>>=pattern.dims().iter().map(|&x|0..x).collect();

	let (c,ix)=compare_subtensors(&data,|d,p,ix|(d.indices().map(|jx|f(&d[&jx],&p[jx])).sum::<f32>(),ix),pattern).min_by(|x,y|x.0.total_cmp(&y.0)).unwrap();
	for n in 0..ix.len(){
		ranges[n].start=ix[n] as usize;
		ranges[n].end +=ix[n] as usize;
	}

	(data.slice(ranges).into_tensor(),c)
}

/// fuzzy finds a table based on the cell pattern
pub fn grab_table_pattern(data:impl AsRef<View<String>>,boolscale:f32,diffscale:f32,levscale:f32,parsescale:f32,pattern:impl AsRef<View<CellPattern>>)->(Tensor<String>,f32){
	grab_table(data,|d,p|p.diff_lev_transform_cost(boolscale,diffscale,levscale,parsescale,d),pattern)
}
/// fuzzy finds a table based on the cell pattern
pub fn grab_table_default(data:impl AsRef<View<String>>,pattern:impl AsRef<View<CellPattern>>)->(Tensor<String>,f32){grab_table_pattern(data,1.0,1.0,1.0,5.0,pattern)}


use b_k_tree::{metrics::Levenshtein,DiscreteMetric};
use crate::{
	builtin_tensor::{GridIter,Position,Tensor,View}
};
use std::{
	cmp::{Ordering,PartialOrd},ops::Range,sync::OnceLock
};
