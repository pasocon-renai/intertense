impl Display for Error{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		let (layout,kind,rhslayout)=(&self.layout,&self.kind,&self.rhslayout);
		let op=self.op;

		{
			writeln!(f,"Error:     {kind}")?;
			writeln!(f,"Operation: {op}")?;
			writeln!(f,"Layout:    {layout:?}")?;
		}
		if let Some(r)=rhslayout{
			writeln!(f,"Rhs:       {r:?}")
		}else{
			writeln!(f,"Rhs:       None")
		}
	}
}
impl Display for ErrorKind{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		match self{
			Self::Alias                             =>write!(f,"mutable components may not alias"),
			Self::BroadcastMismatch                 =>write!(f,"tensor shapes were expected to be broadcast compatible"),
			Self::BufferTooSmall{available,required}=>write!(f,"buffer len is {available} but tensor len is {required}"),
			Self::CountMismatch                     =>write!(f,"tensors were expected to have the same number of components"),
			Self::CountOverflow                     =>write!(f,"component count must not exceed usize::MAX"),
			Self::DimMismatch(ix)                   =>write!(f,"the dims for index {ix} were expected to match"),
			Self::DimOverflow                       =>write!(f,"individual dims may not exceed isize::MAX"),
			Self::EmptyCollection                   =>write!(f,"expected a non empty collection"),
			Self::EmptyTensor                       =>write!(f,"expected a non empty tensor"),
			Self::InvalidIndex(index)               =>write!(f,"index {index} is invalid for this layout"),
			Self::InvalidLayout                     =>write!(f,"layout is invalid"),
			Self::LenOverflow                       =>write!(f,"tensor len must not exceed isize::MAX"),
			Self::Other(e)=>e.fmt(f),
			Self::OutOfBounds(position)                 =>write!(f,"position {position} out of bounds"),
			Self::RankMismatch                          =>write!(f,"tensor ranks were expected to be equal"),
			Self::ShapeMismatch                         =>write!(f,"tensor shapes were expected to be identical"),
			Self::SpecificCountMismatch(c)              =>write!(f,"expected a tensor of with {c} components"),
			Self::SpecificDimMismatch{expectation,index}=>write!(f,"expected the dim of axis {index} to be {expectation}"),
			Self::SpecificRankMismatch(r)               =>write!(f,"expected a tensor of rank {r}"),
			Self::Undefined(s)                          =>write!(f,"operation result is undefined: {s}"),
			Self::Unknown                               =>write!(f,"unknown error")
		}
	}
}
impl Error{
	/// create an aliasing error
	pub fn alias(layout:Layout,op:&'static str)->Self{
		Self::new(layout,ErrorKind::Alias,op,None)
	}
	/// create a broadcast mismatch error
	pub fn broadcast_mismatch(layout:Layout,op:&'static str,rhslayout:Layout)->Self{
		Self::new(layout,ErrorKind::BroadcastMismatch,op,rhslayout)
	}
	/// create a buffer too small error
	pub fn buffer_too_small(layout:Layout,available:usize,required:usize,op:&'static str)->Self{
		Self::new(layout,ErrorKind::BufferTooSmall{available,required},op,None)
	}
	/// create a count mismatch error
	pub fn count_mismatch(layout:Layout,op:&'static str,rhslayout:Layout)->Self{
		Self::new(layout,ErrorKind::CountMismatch,op,rhslayout)
	}
	/// create a count overflow error
	pub fn count_overflow(layout:Layout,op:&'static str)->Self{
		Self::new(layout,ErrorKind::CountOverflow,op,None)
	}
	/// create a dim mismatch error
	pub fn dim_mismatch(layout:Layout,index:isize,op:&'static str,rhslayout:Layout)->Self{
		Self::new(layout,ErrorKind::DimMismatch(index),op,rhslayout)
	}
	/// create a dim overflow error
	pub fn dim_overflow(layout:Layout,op:&'static str)->Self{
		Self::new(layout,ErrorKind::DimOverflow,op,None)
	}
	/// create an emptiness error
	pub fn empty_collection(layout:Layout,op:&'static str)->Self{
		Self::new(layout,ErrorKind::EmptyCollection,op,None)
	}
	/// create an emptiness error
	pub fn empty_tensor(layout:Layout,op:&'static str)->Self{
		Self::new(layout,ErrorKind::EmptyTensor,op,None)
	}
	/// create an invalid index error
	pub fn invalid_index(layout:Layout,index:isize,op:&'static str)->Self{
		Self::new(layout,ErrorKind::InvalidIndex(index),op,None)
	}
	/// create a len overflow error
	pub fn len_overflow(layout:Layout,op:&'static str)->Self{
		Self::new(layout,ErrorKind::LenOverflow,op,None)
	}
	/// create a new error
	pub fn new(layout:Layout,kind:ErrorKind,op:&'static str,rhslayout:impl Into<Option<Layout>>)->Self{
		Self{layout,kind,op,rhslayout:rhslayout.into()}
	}
	/// create another error
	pub fn other<E:'static+StdError>(error:E,layout:Layout,op:&'static str,rhslayout:impl Into<Option<Layout>>)->Self{
		Self::new(layout,ErrorKind::Other(Arc::new(error)),op,rhslayout.into())
	}
	/// create an index error
	pub fn out_of_bounds(layout:Layout,op:&'static str,position:Position)->Self{
		Self::new(layout,ErrorKind::OutOfBounds(position),op,None)
	}
	/// create a rank mismatch error
	pub fn rank_mismatch(layout:Layout,op:&'static str,rhslayout:Layout)->Self{
		Self::new(layout,ErrorKind::RankMismatch,op,rhslayout)
	}
	/// create a shape mismatch error
	pub fn shape_mismatch(layout:Layout,op:&'static str,rhslayout:Layout)->Self{
		Self::new(layout,ErrorKind::ShapeMismatch,op,rhslayout)
	}
	/// create a specific mismatch error
	pub fn specific_count_mismatch(layout:Layout,expectation:usize,op:&'static str)->Self{
		Self::new(layout,ErrorKind::SpecificCountMismatch(expectation),op,None)
	}
	/// create a specific mismatch error
	pub fn specific_dim_mismatch(layout:Layout,expectation:usize,index:isize,op:&'static str)->Self{
		Self::new(layout,ErrorKind::SpecificDimMismatch{expectation,index},op,None)
	}
	/// create a specific mismatch error
	pub fn specific_rank_mismatch(layout:Layout,expectation:usize,op:&'static str)->Self{
		Self::new(layout,ErrorKind::SpecificRankMismatch(expectation),op,None)
	}
	/// create an undefined error
	pub fn undefined(layout:Layout,reason:&'static str,op:&'static str)->Self{
		Self::new(layout,ErrorKind::Undefined(reason),op,None)
	}
	/// create an unknown error. not recommended.
	pub fn unknown(layout:Layout,op:&'static str)->Self{
		Self::new(layout,ErrorKind::Unknown,op,None)
	}

	/// get the error kind
	pub fn get_kind(&self)->ErrorKind{self.kind.clone()}
	/// get the layout
	pub fn get_layout(&self)->Layout{self.layout.clone()}
	/// get the op
	pub fn get_op(&self)->&'static str{self.op}
	/// get the layout
	pub fn get_rhs_layout(&self)->Option<Layout>{self.rhslayout.clone()}

	/// set the layout
	pub fn set_layout(&mut self,layout:Layout){self.layout=layout}
	/// set the op
	pub fn set_op(&mut self,op:&'static str){self.op=op}
	/// set the rhs layout
	pub fn set_rhs_layout(&mut self,rhslayout:impl Into<Option<Layout>>){self.rhslayout=rhslayout.into()}

	/// set the layout
	pub fn with_layout(mut self,layout:Layout)->Self{
		self.set_layout(layout);
		self
	}
	/// set the op
	pub fn with_op(mut self,op:&'static str)->Self{
		self.set_op(op);
		self
	}
	/// set the rhs layout
	pub fn with_rhs_layout(mut self,rhslayout:impl Into<Option<Layout>>)->Self{
		self.set_rhs_layout(rhslayout);
		self
	}
}
impl PartialEq for ErrorKind{
	fn eq(&self,other:&Self)->bool{
		match (self,other){
			(Self::Alias,Self::Alias)=>true,
			(Self::BroadcastMismatch,Self::BroadcastMismatch)=>true,
			(Self::BufferTooSmall{available:ax,required:ix},Self::BufferTooSmall{available:bx,required:jx})=>(ax,ix)==(bx,jx),
			(Self::CountMismatch,Self::CountMismatch)=>true,
			(Self::CountOverflow,Self::CountOverflow)=>true,
			(Self::DimMismatch(a),Self::DimMismatch(b))=>a==b,
			(Self::DimOverflow,Self::DimOverflow)=>true,
			(Self::EmptyCollection,Self::EmptyCollection)=>true,
			(Self::EmptyTensor,Self::EmptyTensor)=>true,
			(Self::InvalidIndex(a),Self::InvalidIndex(b))=>a==b,
			(Self::InvalidLayout,Self::InvalidLayout)=>true,
			(Self::LenOverflow,Self::LenOverflow)=>true,
			(Self::Other(_a),Self::Other(_b))=>false,
			(Self::OutOfBounds(a),Self::OutOfBounds(b))=>*a==*b,
			(Self::RankMismatch,Self::RankMismatch)=>true,
			(Self::ShapeMismatch,Self::ShapeMismatch)=>true,
			(Self::SpecificCountMismatch(a),Self::SpecificCountMismatch(b))=>a==b,
			(Self::SpecificDimMismatch{expectation:ax,index:ix},Self::SpecificDimMismatch{expectation:bx,index:jx})=>(ax,ix)==(bx,jx),
			(Self::SpecificRankMismatch(a),Self::SpecificRankMismatch(b))=>a==b,
			(Self::Undefined(a),Self::Undefined(b))=>a==b,
			(Self::Unknown,Self::Unknown)=>true,
			(_,_)=>false
		}
	}
	fn ne(&self,other:&Self)->bool{
		if let Self::Other(_)=self {return false}
		if let Self::Other(_)=other{return false}

		!self.eq(other)
	}
}
impl StdError for Error{}

#[cfg(test)]
mod tests{
	#[test]
	fn layout(){
		let (mut dims,mut strides)=(Vec::new(),Vec::new());
		// scalar
		assert!(check_layout(&dims,&strides).is_ok());

		dims.push(10);
		let check=check_layout(&dims,&strides).unwrap_err();
		// rank mismatch
		assert_eq!((check.get_kind(),check.get_op()),(ErrorKind::SpecificRankMismatch(1),"check"));

		strides.push(1);
		assert!(check_layout(&dims,&strides).is_ok());

		dims.push(3);
		strides.push(0);
		assert!(check_layout(&dims,&strides).is_ok());
	}
	#[test]
	fn layout_mut(){
		let (mut dims,mut strides)=(Vec::new(),Vec::new());
		// scalar
		assert!(check_layout_mut(&dims,&strides).is_ok());

		dims.push(10);
		let check=check_layout_mut(&dims,&strides).unwrap_err();
		// rank mismatch
		assert_eq!((check.get_kind(),check.get_op()),(ErrorKind::SpecificRankMismatch(1),"check"));

		strides.push(1);
		assert!(check_layout(&dims,&strides).is_ok());

		dims.push(3);
		strides.push(0);
		let check=check_layout_mut(&dims,&strides).unwrap_err();
		assert_eq!((check.get_kind(),check.get_op()),(ErrorKind::Alias,"check"));
	}

	use super::*;
}

#[derive(Clone,Debug,Default)]
#[repr(u8)]
/// tensor operation error
pub enum ErrorKind{
	/// An operation can't proceed because it would require aliasing mutable components between or within tensor or view references
	Alias=1,
	/// The shapes were expected to be broadcast compatible, but they weren't
	BroadcastMismatch,
	/// The tensor doesn't fit in the given buffer
	BufferTooSmall{available:usize,required:usize},
	/// The tensors were expected to contain the same number of components, but they didn't
	CountMismatch,
	/// The component count is greater than usize::MAX
	CountOverflow,
	/// A specific dimension didn't match, but the shapes don't necessarily need to be the same or broadcast compatible
	DimMismatch(isize),
	/// The dim is greater than isize::MAX
	DimOverflow,
	/// A collection of tensors was expected to be non empty
	EmptyCollection,
	/// A tensor shape was expected to be non empty
	EmptyTensor,
	/// A dimension index was out of bounds for a tensor's rank. For example, this could be returned instead of panicking when an operation would access tensor.dims()[6] for a rank 4 tensor
	InvalidIndex(isize),
	/// The layout should be valid at this point
	InvalidLayout,
	/// The length when laid out in a buffer is greater than isize::MAX
	LenOverflow,
	/// Something else happened
	Other(Arc<dyn StdError>),
	/// If the position was outside the tensor
	OutOfBounds(Position),
	/// The ranks were expected to be the same, but they weren't
	RankMismatch,
	/// The shapes were expected to be the same, but they weren't
	ShapeMismatch,
	/// The count was expected to be equal to the specific contained value
	SpecificCountMismatch(usize),
	/// The dim was expected to be a specific value
	SpecificDimMismatch{expectation:usize,index:isize},
	/// The rank was expected to be equal to the specific contained value
	SpecificRankMismatch(usize),
	/// The operation is not defined for a particular input for some reason, such as attempting to invert a non square matrix
	Undefined(&'static str),
	#[default]
	/// Placeholder default value for when the variant has not yet been constructed. Library code should generally avoid returning this.
	Unknown=0,
}
#[derive(Clone,Debug,Default)]
/// tensor operation error
pub struct Error{layout:Layout,kind:ErrorKind,op:&'static str,rhslayout:Option<Layout>}
/// tensor result type
pub type Result<T>=StdResult<T,Error>;

/// check if the position is in bounds. does not check the dims themselves
pub fn check_bounds<P:SignedIndexPosition>(dims:&[usize],position:&[P])->Result<()>{
	let rank=position.len();
	if dims.len()!=rank{return Err(Error::specific_rank_mismatch(diagnostic_shape(dims),rank,"check"))}
											// check the bounds
	for ix in 0..rank{
		let dim=dims[ix] as isize;
		let bounds=-dim..dim;
											// error if px is none or bounds doesn't contain it
		let px=position[ix].try_into().ok();
		if px.map(|px|!bounds.contains(&px)).unwrap_or(true){
			let mut position=Position::try_from_coordinates(position).unwrap_or_default();
											// set the problematic index to the checked value just in case try into is being nondeterministic
			if position.len()>0{
				if let Some(px)=px{position[ix]=px}else{position=Default::default()}
			}
			return Err(Error::out_of_bounds(diagnostic_shape(dims),"check",position))
		}
	}
	Ok(())
}
/// check if dims are broadcast compatible
pub fn check_broadcast(adims:&[usize],bdims:&[usize])->Result<()>{
	let rank=adims.len().max(bdims.len());
	let biter=bdims.iter().copied().rev().chain(iter::repeat(1));
	let aiter=adims.iter().copied().rev().chain(iter::repeat(1));

	for (adim,bdim) in aiter.zip(biter).take(rank){
		if adim!=bdim&&adim!=1&&bdim!=1{return Err(Error::broadcast_mismatch(diagnostic_shape(adims),"check",diagnostic_shape(bdims)))}
	}

	Ok(())
}
/// check the dimensions specifically for dim overflow. does not check strides or anything else
pub fn check_dims(dims:&[usize])->Result<()>{
	for &dim in dims{
		if dim>isize::MAX as usize{return Err(Error::dim_overflow(diagnostic_shape(dims),"check"))}
	}
	Ok(())
}
/// checks if a layout has a valid combination of dims and strides
/// this function verifies:
/// count does not overflow usize
/// dims do not overflow isize
/// the required buffer len does not overflow isize
pub fn check_layout(dims:&[usize],strides:&[isize])->Result<()>{
	//check_dims(dims)         .map_err(|e|e.with_layout(diagnostic_layout(dims,strides)))?; checked count and checked len already check dim overflow
	checked_count(dims)      .map_err(|e|e.with_layout(diagnostic_layout(dims,strides)).with_op("check"))?;
	checked_len(dims,strides).map_err(|e|e.with_layout(diagnostic_layout(dims,strides)).with_op("check"))?;

	Ok(())
}
/// check if a layout has a valid combination of dims and strides and won't alias its components. Rather than enumerating tensor positions, it checks axis's strides and detects overlap using the least common multiple of stride magnitudes.
/// this function verifies:
/// count does not overflow usize
/// dims do not overflow isize
/// the required buffer len does not overflow isize
/// no two positions would map to the same component offset within the tensor buffer
pub fn check_layout_mut(dims:&[usize],strides:&[isize])->Result<()>{
	/// compute the greatest common factor. helper funtion for lcm
	fn gcd(a:usize,b:usize)->usize{
		if a==b{return a}

		let (mut a,mut b)=(a.max(b),a.min(b));
		while b>0{(a,b)=(b,a%b)}

		a
	}
	/// compute the lowest common multiple. helper function for checking stride overlap
	fn lcm(a:usize,b:usize)->usize{a/gcd(a,b)*b}
															// check the layout generally first
	check_layout(dims,strides)?;
															// trivial cases: empty and scalar can't alias
	if dims.contains(&0)||dims.len()==0{return Ok(())}
															// check each axis
	for ix in 0..dims.len(){
															// unsqueeze case: if axis dim is 1, this dim may be skipped since unsqueezing doesn't alias. dim=1 -> only valid indices normalize to 0 -> element not accessible through multiple distinct indices along this axis
		if dims[ix]==1{continue}
															// broadcast case: when an axis overlaps with itself due to broadcasting, the stride is 0. This scenario guarantees alias when dim>1. Axes with dim <= 1 have already been handled above
		if strides[ix]==0{return Err(Error::alias(diagnostic_layout(dims,strides),"check"))}
															// nontrivial case: check against the other axis strides for common stride multiples
		for jx in 0..ix{
			let (ldim,rdim)=(dims[ix],dims[jx]);
			let (lstr,rstr)=(strides[ix].abs() as usize,strides[jx].abs() as usize);
															// skip other axes's unsqueeze case because unsqueezing can't alias
			if rdim==1{continue}
															// compute axis spans and stride lcm
			let (lspan,rspan)=(ldim*lstr,rdim*rstr);
			let firstoverlap=lcm(lstr,rstr);
															// if either span surpasses firstoverlap, a component at firstoverlap is a valid component along that axis. When firstoverlap is a valid offset along both axes, it creates an alias
			if lspan>firstoverlap&&rspan>firstoverlap{
				return Err(Error::alias(diagnostic_layout(dims,strides),"check"))
			}
		}
	}
	Ok(())
}
/// count the components, checking for count or dim overflow
pub fn checked_count(dims:&[usize])->Result<usize>{
	let mut acc:usize=1;
	for &dim in dims{
		let a=acc.checked_mul(dim);

		if dim>isize::MAX as usize{return Err(Error::  dim_overflow(diagnostic_shape(dims),"count"))}
		if a.is_none()            {return Err(Error::count_overflow(diagnostic_shape(dims),"count"))}

		acc=a.unwrap();
	}

	Ok(acc)
}
/// count the len, checking for dim or len overflow, and for mismatched dim/stride rank.
pub fn checked_len(dims:&[usize],strides:&[isize])->Result<usize>{
	let rank=checked_rank(dims,strides).map_err(|e|e.with_op("len"))?;
	if dims.contains(&0){
		return check_dims(dims).map_err(|e|e.with_layout(diagnostic_layout(dims,strides)).with_op("len")).map(|_|0)
	}
							// each axis independently contributes (dim-1)*|stride| to the maximum offset. The buffer length will be 1 more than the maximum
	let mut acc:usize=1;
	for ix in 0..rank{
		let (dim,stride)=(dims[ix],strides[ix]);
		if dim>isize::MAX as usize{return Err(Error::dim_overflow(diagnostic_layout(dims,strides),"len"))}
							// saturate so we don't accidently end up below isize::MAX due to being above usize::MAX rather than genuinely not overflowing
		acc=acc.saturating_add((dim-1).saturating_mul(stride.abs() as usize));
		if acc>isize::MAX as usize{return Err(Error::len_overflow(diagnostic_layout(dims,strides),"len"))}
	}

	Ok(acc)
}
/// find the rank of a layout, returning a rank mismatch if the dims and strides have mismatched lengths
pub fn checked_rank(dims:&[usize],strides:&[isize])->Result<usize>{
	let (r,s)=(dims.len(),strides.len());
	if r!=s{return Err(Error::specific_rank_mismatch(diagnostic_layout(dims,strides),r.max(s),"rank"))}

	Ok(r)
}
/// check if dims are exact match compatible
pub fn check_shape_match(adims:&[usize],bdims:&[usize])->Result<()>{
	if adims!=bdims{return Err(Error::shape_mismatch(diagnostic_shape(adims),"check",diagnostic_shape(bdims)))}
	Ok(())
}
/// create a layout from slices quickly without validation, for diagnostic purposes
pub fn diagnostic_layout(dims:&[usize],strides:&[isize])->Layout{Layout::from_inner(dims.to_vec(),strides.to_vec())}
/// create a layout that lacks strides for diagnostic purposes specifically relating to dimensions
pub fn diagnostic_shape(dims:&[usize])->Layout{Layout::from_inner(dims.to_vec(),Vec::new())}

use std::{
	cmp::PartialEq,error::Error as StdError,fmt::{Display,Formatter,Result as FmtResult},iter,result::Result as StdResult,sync::Arc
};
use super::{
	position::{Position,SignedIndexPosition},layout::Layout
};
