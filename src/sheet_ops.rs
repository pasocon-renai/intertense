
//#[track_caller]
/// search for a component's position within a tensor view, then access another component in another view with that position
pub fn xlookup<'a,E:Display,X:Display+PartialOrd<E>+PartialOrd<X>>(lookupquery:&E,lookupvalues:&View<X>,returnvalues:&'a View<X>,notfound:impl Into<Option<&'a X>>,mmode:impl Into<Option<i32>>,smode:impl Into<Option<i32>>)->Option<&'a X>{xmatch(lookupquery,lookupvalues,mmode,smode).map(|ix|&returnvalues[ix]).or_else(||notfound.into())}
//#[track_caller]
/// search for a component's position within a tensor view. Supported m modes: -1=find query's floor, 0 (default)=exact (==), 1=find query's ceil. Supported s modes: -1=reverse, 1 (default)=forward, 2=binary ascending, -2=binary descending. //TODO m mode -2=wildcard (?=single character, *=any sequence, ~=?, *, or ~)
pub fn xmatch <   E:Display,X:Display+PartialOrd<E>+PartialOrd<X>>(lookupquery:&E,lookupvalues:&View<X>,mmode:impl Into<Option<i32>>,smode:impl Into<Option<i32>>)->Option<Position>{
	let (binary,reverse)=match smode.into().unwrap_or(1){
		-1=>(false,true),1=>(false,false),
		-2=>(true ,true),2=>(true ,false),
		x=>panic!("invalid search mode {x}. search mode must be +-1 or +-2")
	};
	let ceilfloorcandidate=match mmode.into().unwrap_or(0){
		-1=>Some(Ordering::Less),1=>Some(Ordering::Greater),
		-2=>todo!(),
		0 =>None,
		x=>panic!("invalid match mode {x}. match mode currently must be +-1 or 0. -2 could theoretically be allowed but isn't yet supported")
	};
	let dims=lookupvalues.dims();
	let rank=dims.len();

	if dims.contains(&0){return None}

	if binary{
		let ceilfloorcandidate=if reverse{ceilfloorcandidate.map(|r|r.reverse())}else{ceilfloorcandidate};
		match match_tensor::binary_search_by(|ix|{
			let r=lookupvalues[ix].partial_cmp(lookupquery);
			r.map(|r|if reverse{r.reverse()}else{r})
		},dims){
			Err(mut ix)=>if ix[0]==dims[0] as isize{
				if ceilfloorcandidate==Some(Ordering::Less){
					for n in 0..rank{ix[n]=(dims[n]-1) as isize}
					Some(ix)
				}else{
					None
				}
			}else{
				if ceilfloorcandidate==Some(Ordering::Less){
					ix.decrement(dims);
				}    {
					Some(ix)
				}
			},
			Ok(ix) =>Some(ix)
		}
	}else{
		let mut best:Option<(&X,Position)>=None;
		for ix in (!reverse).then(||lookupvalues.positions()).into_iter().flatten().chain(reverse.then(||lookupvalues.positions().rev()).into_iter().flatten()){
			let r=lookupvalues[&ix].partial_cmp(lookupquery);

			if r==None{continue}
			if r==Some(Ordering::Equal){return Some(ix)}
			if r==ceilfloorcandidate{
				let c=&lookupvalues[&ix];
				if best.as_ref().map(|&(b,ref _jx)|b.partial_cmp(c)==ceilfloorcandidate).unwrap_or(true){best=Some((c,ix))}
			}
		}
		best.map(|(_b,jx)|jx)
	}
}


use crate::{
	builtin_tensor::{Position,View},match_tensor
};
use std::{
	cmp::{Ordering,PartialOrd},fmt::Display
};
