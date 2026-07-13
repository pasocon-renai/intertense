impl AsMut<Layout> for Layout{
	fn as_mut(&mut self)->&mut Layout{self}
}
impl AsRef<Layout> for Layout{
	fn as_ref(&self)->&Layout{self}
}
impl Layout{
	#[track_caller]
	/// broadcast a specific axis. panics if the dims would be incompatible, or if dims and strides have mismatched lengths. panics if the index is out of bounds
	pub fn broadcast_dim(&mut self,index:impl SignedIndexPosition,rhs:usize)->&mut Self{
		assert_eq!(self.dims().len(),self.strides().len());

		let rank=self.dims().len();
		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};

		let (dims,strides)=self.inner_mut();
		if rhs!=1{
			if dims[index]!=1&&dims[index]!=rhs{panic!("dim at {index} must be either 1 or {rhs} to broadcast to {rhs}. dim: {}",dims[index])}

			dims[index]   =rhs;
			strides[index]=0;
		}

		self
	}
	/// computes the offset of a component given its coordinates
	pub fn compute_offset<I:SignedIndexPosition>(&self,position:&[I])->usize{position::compute_offset(self.dims(),position,self.strides())}
	/// counts the number of components. invalid layouts may return something unhelpful or unexpected
	pub fn count(&self)->usize{self.dims().iter().product()}
	/// references the dims
	pub fn dims(&self)->&Vec<usize>{&self.inner.0}
	/// references the dims. note that most nontrivial functionality expects the lengths of dims and strides to be equal, and for their combination to form a valid layout. violating this invariant may result in panics or unexpected behavior
	pub fn dims_mut(&mut self)->&mut Vec<usize>{&mut Arc::make_mut(&mut self.inner).0}
	#[track_caller]
	/// reverse the order of components along all axes except the one at the index. panics if the index is out of bounds
	pub fn flip_around(&mut self,index:impl SignedIndexPosition)->&mut Self{
		let strides=self.strides_mut();
		let rank=strides.len();

		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};
		strides[index]*=-1;

		self.flip()
	}
	#[track_caller]
	/// reverse the order of components along the axis. panics if the index is out of bounds
	pub fn flip_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		let rank=self.strides().len();
		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};

		self.strides_mut()[index]*=-1;
		self
	}
	/// reverse the order of components along all axes
	pub fn flip(&mut self)->&mut Self{
		for s in self.strides_mut().iter_mut(){*s*=-1}
		self
	}
	/// creates a layout from the dims and strides. does not check validity
	pub fn from_inner(dims:Vec<usize>,strides:Vec<isize>)->Self{
		Self{inner:Arc::from((dims,strides))}
	}
	#[track_caller]
	/// get the dimension of the corresponding index. panics if the index is out of bounds
	pub fn get_dim(&self,index:impl SignedIndexPosition)->usize{
		let dims=self.dims();
		let rank=dims.len();

		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};
		dims[index]
	}
	#[track_caller]
	/// get the stride of the corresponding index. panics if out of bounds
	pub fn get_stride(&self,index:impl SignedIndexPosition)->isize{
		let strides=self.strides();
		let rank=strides.len();

		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};
		strides[index]
	}
	/// references the dims. note that most nontrivial functionality expects the lengths of dims and strides to be equal, and for their combination to form a valid layout. violating this invariant may result in panics or unexpected behavior
	pub fn inner_mut(&mut self)->(&mut Vec<usize>,&mut Vec<isize>){
		let (d,s)=Arc::make_mut(&mut self.inner);
		(d,s)
	}
	/// converts into the dims
	pub fn into_dims(self)->Vec<usize>{
		if Arc::strong_count(&self.inner)==1{
			Arc::into_inner(self.inner).unwrap().0
		}else{
			self.dims().to_vec()
		}
	}
	/// convert into the inner value
	pub fn into_inner(self)->(Vec<usize>,Vec<isize>){
		if Arc::strong_count(&self.inner)==1{
			Arc::into_inner(self.inner).unwrap()
		}else{
			(self.dims().to_vec(),self.strides().to_vec())
		}
	}
	/// checks if the dims contain 0
	pub fn is_empty(&self)->bool{self.dims().contains(&0)}
	/// checks if the layout is normalized, returning true if it is and false if it isn't. also returns false if the layout is invalid. A layout is considered normalized if and only if mapping a lexiconographic iteration over the indices to components would iterate over the same components in the same order as iterating over the underlying buffer. Normalizing a layout may require normalizing the buffer too.
	pub fn is_normalized(&self)->bool{
		let mut acc:usize=1;
		if self.dims().len()!=self.strides().len(){return false}
									// the normalization condition generally equivalent to whether each stride is equal to the product of all the dims after it
		for (&d,&s) in self.dims().iter().rev().zip(self.strides().iter().rev()){
			if acc as isize!=s{return false}
			acc=acc.saturating_mul(d);
									// if the product of dims is greater than isize::MAX, the only way this could be a valid layout is with a broadcast, which would not be considered normalized
			if acc>isize::MAX as usize{return false}
		}
		true
	}
	/// check if this layout represents a scalar. We use dims.len()==0 to represent scalars
	pub fn is_scalar(&self)->bool{self.dims().is_empty()}
	/// computes the length of a slice required to store a tensor with this layout. the backing vec of a tensor will store the data in a slice of this length starting at index offset in a vec of at least offset+len. May return something unexpected or unhelpful if the layout is invalid
	pub fn len(&self)->usize{position::buffer_len(self.dims(),self.strides())}
	/// creates a new tensor layout from the dimensions
	pub fn new<D:AsRef<[usize]>>(dims:D)->Self{
		match Self::try_new(dims){
			Err(e)=>panic!("{e}"),
			Ok(x)=>x
		}
	}
	/// iterate over the positions in the tensor. The iterator will iterate once over every position in bounds of dims, with positive positions from forward iteration and negative positions from reverse iteration. panics if the dims are invalid (if any exceed isize::MAX, or their product overflows a usize)
	pub fn positions(&self)->PositionIter{PositionIter::new(self.dims())}
	/// returns the number of dims in the tensor
	pub fn rank(&self)->usize{self.dims().len()}
	#[track_caller]
	/// slice dim
	pub fn slice_dim<I:SignedIndexPosition>(&mut self,index:impl SignedIndexPosition,offset:&mut usize,range:impl RangeBounds<I>)->&mut Self{
		let (dim,stride)=(self.get_dim(index),self.get_stride(index));
		let start=match range.start_bound(){
			Bound::Excluded(&px)=>position::unsign_range_bound(dim,px).expect("range should be in bounds of the dim")+1,
			Bound::Included(&px)=>position::unsign_range_bound(dim,px).expect("range should be in bounds of the dim"),
			Bound::Unbounded=>0
		};
		let stop=match range.end_bound(){
			Bound::Excluded(&px)=>position::unsign_range_bound(dim,px).expect("range should be in bounds of the dim"),
			Bound::Included(&px)=>position::unsign_range_bound(dim,px).expect("range should be in bounds of the dim")+1,
			Bound::Unbounded=>dim
		};

		let dims=self.dims_mut();
		let index=position::unsign_index(index,dims.len()).unwrap();

		dims[index]=stop-start;
		*offset+=if stride<0{dim-start}else{start}*stride.abs() as usize;

		self
	}
	#[track_caller]
	/// compute the layout resulting from a slice operation
	pub fn slice<I:SignedIndexPosition,R:RangeBounds<I>>(&mut self,offset:&mut usize,ranges:&[R])->&mut Self{
		match Self::try_slice(self,offset,ranges){
			Err(e)=>panic!("{e}"),
			Ok(x)=>x
		}
	}
	#[track_caller]
	/// squeeze an axis of dim 1 into nonexistence. panics if the dim at the index is not equal to 1. panics if out of bounds of the rank on the basis of symmetry with unsqueeze
	pub fn squeeze_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		assert_eq!(self.dims().len(),self.strides().len());

		let rank=self.dims().len();
		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};

		let (dims,strides)=self.inner_mut();
		assert_eq!(dims[index],1);

		dims.remove(index);
		strides.remove(index);

		self
	}
	/// references the strides
	pub fn strides(&self)->&Vec<isize>{&self.inner.1}
	/// references the strides. note that most nontrivial functionality expects the lengths of dims and strides to be equal, and for their combination to form a valid layout. violating this invariant may result in panics or unexpected behavior
	pub fn strides_mut(&mut self)->&mut Vec<isize>{&mut Arc::make_mut(&mut self.inner).1}
	#[track_caller]
	/// swap a pair of axes
	pub fn swap_dims(&mut self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->&mut Self{
		assert_eq!(self.dims().len(),self.strides().len());

		let rank=self.dims().len();
		let (ax,bx)=(if let Some(ix)=position::unsign_index(a,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",a.expect_isize("must be able to convert index to isize"))},if let Some(ix)=position::unsign_index(b,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",b.expect_isize("must be able to convert index to isize"))});

		self.dims_mut   ().swap(ax,bx);
		self.strides_mut().swap(ax,bx);
		self
	}
	/// try broadcasting the dims, returning an error if the dims are not broadcast compatible with rhs. does not explicitly validate either layout. the result for invalid layouts with broadcast compatible dims is unspecified
	pub fn try_broadcast<D:AsRef<[usize]>>(&mut self,rhs:D)->Result<&mut Self>{
		let rhs=rhs.as_ref();
		error::check_broadcast(self.dims(),rhs).map_err(|e|e.with_layout(self.clone()).with_op("broadcast"))?;

		let (dims,strides)=self.inner_mut();

		if dims   .len()<rhs.len(){*dims   =iter::repeat(1).take(rhs.len()-dims   .len()).chain(mem::take(dims   )).collect()}
		if strides.len()<rhs.len(){*strides=iter::repeat(0).take(rhs.len()-strides.len()).chain(mem::take(strides)).collect()}

		for ((d,s),r) in dims.iter_mut().rev().zip(strides.iter_mut().rev()).zip(rhs.iter().rev()){
			if *r==1{continue}
			if *d==1{
				*d=*r;
				*s=0;
			}
		}
		Ok(self)
	}
	/// try to create a new layout, returning an error if the dims are invalid
	pub fn try_new<D:AsRef<[usize]>>(dims:D)->Result<Self>{
		let dims=dims.as_ref();
		error::check_dims(dims).map_err(|e|e.with_op("new"))?;

		let mut acc:usize=1;
		let dims=dims.to_vec();
		let rank=dims.len();
		let mut strides=vec![0;rank];

		for ix in (0..rank).rev(){
			strides[ix]=acc as isize;
			acc*=dims[ix];
		}
		Ok(Self::from_inner(dims,strides))
	}
	/// try to compute the layout resulting from a slice operation, returning an error if the ranges are out of bounds of the dims+1, or if any range start is beyond its stop. does not validate the layout itself. The offset will only be accumulated if the result is Ok
	pub fn try_slice<I:SignedIndexPosition,R:RangeBounds<I>>(&mut self,offset:&mut usize,ranges:&[R])->Result<&mut Self>{
		let (dims,strides)=(self.dims(),self.strides());
		let rank=ranges.len();
																// check rank
		if dims   .len()!=rank{return Err(Error::specific_rank_mismatch(self.clone(),rank,"slice"))}
		if strides.len()!=rank{return Err(Error::specific_rank_mismatch(self.clone(),rank,"slice"))}
																// create temp vectors which can later be reused as new dims and strides for the subview
		let mut newdims :Vec<usize>=vec![0;rank];
		let mut position:Vec<isize>=vec![0;rank];
																// check end bound
		for ix in 0..rank{
			let dim=dims[ix];
			let px=match ranges[ix].end_bound().map(|&x|x.try_into()){
				Bound::Excluded(Ok(px))=>px,
				Bound::Included(Ok(px))=>if px==-1{dim as isize}else{px+1},
				Bound::Unbounded=>dim as isize,
				_=>return Err(Error::out_of_bounds(self.clone(),"slice",Default::default()))
			};

			newdims [ix]=dim+1;
			position[ix]=px;
		}
		error::check_bounds(&newdims,&position).map_err(|e|e.with_layout(self.clone()).with_op("slice"))?;
																// check start bound
		for ix in 0..rank{
			let dim=dims[ix];
			let px=match ranges[ix].start_bound().map(|&x|x.try_into()){
				Bound::Excluded(Ok(px))=>if px==-1{dim as isize}else{px+1},
				Bound::Included(Ok(px))=>px,
				Bound::Unbounded=>dim as isize,
				_=>return Err(Error::out_of_bounds(self.clone(),"slice",Default::default()))
			};

			newdims [ix]=position::unsign_range_bound(dim,position[ix]).unwrap()+1;
			position[ix]=px;
		}
		error::check_bounds(&newdims,&position).map_err(|e|e.with_layout(self.clone()).with_op("slice"))?;
																// now that bounds are checked correct newdims to the dims of the resulting layout
		for ix in 0..rank{newdims[ix]-=position::unsign_range_bound(dims[ix],position[ix]).unwrap()+1}
		*offset+=position::compute_offset(dims,&position,strides).clamp(0,self.len());
																// create new layout
		position.copy_from_slice(strides);
		*self=Self::from_inner(newdims,position);

		Ok(self)
	}
	/// unsqueeze an axis of dim 1 into existence and insert it at the index. panics if the index is out of bounds of the new rank
	pub fn unsqueeze_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		assert_eq!(self.dims().len(),self.strides().len());

		let rank=self.dims().len()+1;
		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};

		let (dims,strides)=self.inner_mut();

		dims.insert(index,1);
		strides.insert(index,0);

		self
	}
	/// checks if a layout has a valid combination of dims and strides
	/// this function verifies:
	/// dims and strides have matching ranks
	/// count does not overflow usize
	/// dims do not overflow isize
	/// the required buffer len does not overflow isize
	/// the required buffer len is within bufferlen
	pub fn validate(&self,bufferlen:usize)->Result<()>{
		error::check_layout(self.dims(),self.strides()).map_err(|e|e.with_op("validate"))?;
		let len=self.len();

		if bufferlen<len{return Err(Error::buffer_too_small(self.clone(),bufferlen,len,"validate"))}
		Ok(())
	}
	/// check if a layout has a valid combination of dims and strides and won't alias its components. Rather than enumerating tensor positions, it checks axis's strides and detects overlap using the least common multiple of stride magnitudes.
	/// this function verifies:
	/// dims and strides have matching ranks
	/// count does not overflow usize
	/// dims do not overflow isize
	/// the required buffer len does not overflow isize
	/// the required buffer len is within bufferlen
	/// no two positions would map to the same component offset within the tensor buffer
	pub fn validate_mut(&self,bufferlen:usize)->Result<()>{
		error::check_layout_mut(self.dims(),self.strides()).map_err(|e|e.with_layout(self.clone()).with_op("validate"))?;
		let len=self.len();

		if bufferlen<len{return Err(Error::buffer_too_small(self.clone(),bufferlen,len,"validate"))}
		Ok(())
	}
}

#[cfg(feature="serial")]
mod serial{
	impl<'a> Deserialize<'a> for Layout{
		fn deserialize<D:Deserializer<'a>>(deserializer:D)->StdResult<Self,D::Error>{
			match Version::deserialize(deserializer){
				Ok(Version::Current(inner))=>Ok(Layout{inner}),
				Ok(Version::Old{dims,strides,..})=>Ok(Layout::from_inner(dims,strides)),
				Err(e)=>Err(e)
			}
		}
	}
	impl Serialize for Layout{
		fn serialize<S:Serializer>(&self,serializer:S)->StdResult<S::Ok,S::Error>{Version::Current(self.inner.clone()).serialize(serializer)}
	}

	#[derive(Deserialize,Serialize)]
	#[serde(untagged)]
	/// serial layout structure to ensure compatibility with legacy files
	enum Version{
		Current(Arc<(Vec<usize>,Vec<isize>)>),
		Old{dims:Vec<usize>,stray:bool,strides:Vec<isize>}
	}

	use serde::{Deserialize,Deserializer,Serialize,Serializer};
	use std::result::Result as StdResult;
	use super::*;
}

#[derive(Clone,Debug,Default,Eq,Hash,PartialEq)]
#[repr(transparent)]
/// dimensions and strides for how a tensor data is layed out in memory
pub struct Layout{inner:Arc<(Vec<usize>,Vec<isize>)>}

use std::{
	iter,mem,ops::{Bound,RangeBounds},sync::Arc
};
use super::{
	Error,Result,error,position::{PositionIter,SignedIndexPosition,self}
};
