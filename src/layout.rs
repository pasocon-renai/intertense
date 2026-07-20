impl AsMut<Layout> for Layout{
	fn as_mut(&mut self)->&mut Layout{self}
}
impl AsRef<Layout> for Layout{
	fn as_ref(&self)->&Layout{self}
}/*
impl Default for Validity{
	fn default()->Self{Self::Mut(0)}
}*/
impl Layout{
	#[track_caller]
	/// broadcast a specific axis. Panics if the current or resulting layout is invalid, or if the index is out of bounds.
	/// If the dimension at the index is 1 and rhs!=1, the axis's dim and stride are set to (rhs, 0). If rhs==1, the dim and stride are unaffected.
	pub fn broadcast_dim(&mut self,index:impl SignedIndexPosition,rhs:usize)->&mut Self{
		error::unwrap_or_panic(self.check().map_err(|e|e.with_op("broadcast")));
									// normalize index before checking rhs to always panic if the index is out of bounds. Ensure reuse of the normalized index as opposed to the original to guard against poorly behaved TryFrom implementations
		let index=self.normalize_index(index);
		if rhs==1{return self}
									// the layout could be invalidated by the expansion of the count, or by rhs itself not being a valid dim.
		let dim=self.dims()[index];
		if dim==1{
			assert!(rhs<=isize::MAX as usize);
			assert!(self.count().checked_mul(rhs).is_some());

			self.   dims_mut()[index]=rhs;
			self.strides_mut()[index]=0;
		}else if dim!=rhs{
			panic!("cannot broadcast unequal dims unless one of them is equal to 1. left: {dim} right: {rhs}")
		}

		self
	}
	#[track_caller]
	/// try broadcasting the dims, panicingif the dims are not broadcast compatible with rhs. does not explicitly validate either layout. the result for invalid layouts with broadcast compatible dims is unspecified
	pub fn broadcast<D:AsRef<[usize]>>(&mut self,rhs:D)->&mut Self{error::unwrap_or_panic(self.try_broadcast(rhs))}
	#[track_caller]
	/// computes the linear buffer offset for the position. Panics if position is out of bounds.
	pub fn compute_offset<I:SignedIndexPosition>(&self,position:&[I])->usize{position::compute_offset(self.dims(),position,self.strides())}
	/// counts the number of components. The result is only meaningful for valid layouts. To produce an Error if the count overflows, use error::checked_count
	pub fn count(&self)->usize{self.dims().iter().product()}
	/// references the dims
	pub fn dims(&self)->&[usize]{&self.inner.0}
	/// references the dims. Note that setting dims or strides may affect the validity of the layout.
	pub fn dims_mut(&mut self)->&mut [usize]{&mut Arc::make_mut(&mut self.inner).0}
	#[track_caller]
	/// reverse the order of components along all axes except the one at the index. panics if the index is out of bounds
	pub fn flip_around(&mut self,index:impl SignedIndexPosition)->&mut Self{self.flip_dim(index).flip()}
	#[track_caller]
	/// reverse the order of components along the axis. panics if the index is out of bounds
	pub fn flip_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		let index=self.normalize_index(index);

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
	/// get the dimension of the corresponding axis. panics if the index is out of bounds
	pub fn get_dim(&self,index:impl SignedIndexPosition)->usize{self.dims()[self.normalize_index(index)]}
	#[track_caller]
	/// set the stride of the corresponding axis. panics if out of bounds. Note that setting dims or strides may affect the validity of the layout.
	pub fn get_stride(&self,index:impl SignedIndexPosition)->isize{self.strides()[self.normalize_index(index)]}
	#[track_caller]
	/// insert a new axis at the index. panics if the dim exceeds isize::MAX, or if the index is out of bounds of rank+1. This is operation may affect layout validity.
	pub fn insert_axis(&mut self,index:impl SignedIndexPosition,dim:usize,stride:isize){
		let rank=self.rank()+1;
		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};
									// Ensure reuse of the normalized index as opposed to the original to guard against poorly behaved TryFrom implementations
		self.insert_dim(index,dim);
		self.insert_stride(index,stride);
	}
	#[track_caller]
	/// insert a dim at the index. panics if the dim exceeds isize::MAX, or if the index is out of bounds of rank+1. This is a low level operation that affects layout validity; callers should consider using unsqueeze_dim and broadcast_dim, or insert_axis instead.
	pub fn insert_dim(&mut self,index:impl SignedIndexPosition,dim:usize){
		let rank=self.rank()+1;
		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};
									// Ensure reuse of the normalized index as opposed to the original to guard against poorly behaved TryFrom implementations
		Arc::make_mut(&mut self.inner).0.insert(index,dim);
	}
	#[track_caller]
	/// insert a stride at the index. panics if the index is out of bounds of rank+1. This is a low level operation that affects layout validity; callers should consider using unsqueeze_dim and broadcast_dim, or insert_axis instead.
	pub fn insert_stride(&mut self,index:impl SignedIndexPosition,stride:isize){
		let rank=self.rank()+1;
		let index=if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))};
									// Ensure reuse of the normalized index as opposed to the original to guard against poorly behaved TryFrom implementations
		Arc::make_mut(&mut self.inner).1.insert(index,stride)
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
	/// Computes the minimum buffer length required to store a tensor with this layout. The result is only meaningful for valid layouts. To produce an Error if the len overflows or if dims and strides have mismatched rank, use error::checked_len.
	pub fn len(&self)->usize{position::buffer_len(self.dims(),self.strides())}
	#[track_caller]
	/// creates a new tensor layout from the dimensions
	pub fn new<D:AsRef<[usize]>>(dims:D)->Self{error::unwrap_or_panic(Self::try_new(dims))}
	#[track_caller]
	/// Normalizes a signed index into the range `0..rank`. Panics if the index cannot be represented as `isize` or lies outside the valid signed range `[-rank, rank)`.
	pub fn normalize_index(&self,index:impl SignedIndexPosition)->usize{
		let rank=self.rank();
		if let Some(ix)=position::unsign_index(index,rank){ix}else{panic!("index {} is out of bounds for rank {rank}",index.expect_isize("must be able to convert index to isize"))}
	}
	#[track_caller]
	/// creates a new position iter over the specified dimensions. Panics if the dims are invalid (if any exceed isize::MAX, or their product overflows a usize)
	/// The iterator visits every in-bounds position exactly once. Forward iteration begins at the zero position, while reverse iteration begins at the all -1 position. Though the coordinate data may not be identical, forward and reverse iteration produce an equivalent set of positions according to the signed indexing convention. Note that setting the bounds may affect the signs of the yieled position coordinates
	pub fn positions(&self)->PositionIter{error::unwrap_or_panic(self.clone().try_into())}
	#[track_caller]
	/// insert a new axis at the index. panics if the dim exceeds isize::MAX. This is operation may affect layout validity.
	pub fn push_axis(&mut self,dim:usize,stride:isize){
		self.push_dim(dim);
		self.push_stride(stride);
	}
	#[track_caller]
	/// push a dim onto the layout. panics if index is out of bounds or dim exceeds isize::MAX. This is a low level operation that affects layout validity; callers should consider using unsqueeze_dim and broadcast_dim, or push_axis instead.
	pub fn push_dim(&mut self,dim:usize){
		assert!(dim<=isize::MAX as usize);

		Arc::make_mut(&mut self.inner).0.push(dim);
	}
	#[track_caller]
	/// push a stride onto the layout. This is a low level operation that affects layout validity; callers should consider using unsqueeze_dim and broadcast_dim, or push_axis instead.
	pub fn push_stride(&mut self,stride:isize){Arc::make_mut(&mut self.inner).1.push(stride)}
	/// returns the number of axes in the tensor
	pub fn rank(&self)->usize{self.dims().len()}
	#[track_caller]
	/// remove an axis at the index. panics if the index is out of bounds. This is operation may affect layout validity.
	pub fn remove_axis(&mut self,index:impl SignedIndexPosition)->(usize,isize){(self.remove_dim(index),self.remove_stride(index))}
	#[track_caller]
	/// remove a dim at the index. panics if the index is out of bounds. This is a low level operation that affects layout validity; callers should consider using squeeze_dim or remove_axis instead.
	pub fn remove_dim(&mut self,index:impl SignedIndexPosition)->usize{
		let index=self.normalize_index(index);

		Arc::make_mut(&mut self.inner).0.remove(index)
	}
	#[track_caller]
	/// remove a dim at the index. panics if the index is out of bounds. This is a low level operation that affects layout validity; callers should consider using squeeze_dim or remove_axis instead.
	pub fn remove_stride(&mut self,index:impl SignedIndexPosition)->isize{
		let index=self.normalize_index(index);

		Arc::make_mut(&mut self.inner).1.remove(index)
	}
	/// create a layout for a scalar
	pub fn scalar()->Self{Self::from_inner(Vec::new(),Vec::new())}
	#[track_caller]
	/// set the dimension of the corresponding axis. panics if the dim exceeds isize::MAX or if the index is out of bounds. Note that setting dims or strides may affect the validity of the layout.
	pub fn set_dim(&mut self,index:impl SignedIndexPosition,dim:usize){
		assert!(dim<=isize::MAX as usize);
		let index=self.normalize_index(index);

		Arc::make_mut(&mut self.inner).0[index]=dim
	}
	#[track_caller]
	/// set the stride of the corresponding axis. panics if the index is out of bounds. Note that setting dims or strides may affect the validity of the layout.
	pub fn set_stride(&mut self,index:impl SignedIndexPosition,stride:isize){
		let index=self.normalize_index(index);
		Arc::make_mut(&mut self.inner).1[index]=stride
	}
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
	pub fn slice<I:SignedIndexPosition,R:RangeBounds<I>>(&mut self,offset:&mut usize,ranges:&[R])->&mut Self{error::unwrap_or_panic(self.try_slice(offset,ranges))}
	#[track_caller]
	/// squeeze an axis of dim 1 into nonexistence. panics if the dim at the index is not equal to 1. panics if out of bounds of the rank
	pub fn squeeze_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		error::unwrap_or_panic(self.check());
									// normalize index before checking rhs to always panic if the index is out of bounds. Ensure reuse of the normalized index as opposed to the original to guard against poorly behaved TryFrom implementations
		let index=self.normalize_index(index);
		assert_eq!(self.dims()[index],1);

		self.remove_axis(index);
		self
	}
	/// references the strides
	pub fn strides(&self)->&[isize]{&self.inner.1}
	/// references the strides. Note that setting dims or strides may affect the validity of the layout.
	pub fn strides_mut(&mut self)->&mut [isize]{&mut Arc::make_mut(&mut self.inner).1}
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
	/// try broadcasting the dims, returning an error if the dims are not broadcast compatible with rhs
	pub fn try_broadcast<D:AsRef<[usize]>>(&mut self,rhs:D)->Result<&mut Self>{
		self.check().map_err(|e|e.with_op("broadcast"))?;

		let rhs=rhs.as_ref();
		if self.rank()!=rhs.len(){return Err(Error::rank_mismatch(self.clone(),"broadcast",error::diagnostic_shape(rhs)))}

		let (mut dims,mut strides)=self.clone().into_inner();
		for ix in 0..rhs.len(){
			let dim=dims[ix];
			let rdm=rhs [ix];

			if rdm==1{continue}
			if dim==1{
				dims   [ix]=rdm;
				strides[ix]=0;
			}else if dim!=rdm{
				return Err(Error::broadcast_mismatch(self.clone(),"broadcast",error::diagnostic_shape(rhs)));
			}
		}

		let result=Layout::from_inner(dims,strides);
		result.check().map_err(|e|e.with_op("broadcast"))?;

		*self=result;
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
	/// Normalizes a signed index into the range `0..rank`. Returns Err if the index cannot be represented as `isize` or lies outside the valid signed range `[-rank, rank)`.
	/// Currently, the Error type lacks a special case for if the index could not be converted to isize, so isize::MIN is used to populate the error's index field in that case.
	pub fn try_normalize_index(&self,index:impl SignedIndexPosition)->Result<usize>{
		let rank=self.rank();
		let index=index.try_into().unwrap_or(isize::MIN);

		position::unsign_index(index,rank).ok_or_else(||Error::invalid_index(self.clone(),index,"unsign"))
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
	#[track_caller]
	/// unsqueeze an axis of dim 1 into existence and insert it at the index. panics if the index is out of bounds of the new rank
	pub fn unsqueeze_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		error::unwrap_or_panic(self.check().map_err(|e|e.with_op("unsqueeze")));

		self.insert_axis(index,1,1);
		self
	}

	/// checks if a layout has a valid combination of dims and strides
	/// this function verifies:
	/// dims and strides have matching ranks
	/// count does not overflow usize
	/// dims do not overflow isize
	/// the required buffer len does not overflow isize
	pub fn check(&self)->Result<()>{error::check_layout(self.dims(),self.strides()).map_err(|e|e.with_layout(self.clone()))}
	/// check if a layout has a valid combination of dims and strides and won't alias its components. Rather than enumerating tensor positions, it checks axis's strides and detects overlap using the least common multiple of stride magnitudes.
	/// this function verifies:
	/// dims and strides have matching ranks
	/// count does not overflow usize
	/// dims do not overflow isize
	/// the required buffer len does not overflow isize
	/// the required buffer len is within bufferlen
	/// no two positions would map to the same component offset within the tensor buffer
	pub fn check_mut(&self)->Result<()>{error::check_layout_mut(self.dims(),self.strides()).map_err(|e|e.with_layout(self.clone()))}
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
	/// create a layout for a rank 1 tensor
	pub fn vector(len:usize)->Self{Self::from_inner(vec![len],vec![1])}
}

#[cfg(feature="serial")]
mod serial{
	impl<'a> Deserialize<'a> for Layout{
		fn deserialize<D:Deserializer<'a>>(deserializer:D)->StdResult<Self,D::Error>{
			match SerialLayout::deserialize(deserializer){
				Ok(SerialLayout::Current(inner))=>Ok(Layout{inner}),
				Ok(SerialLayout::Old{dims,strides,..})=>Ok(Layout::from_inner(dims,strides)),
				Err(e)=>Err(e)
			}
		}
	}
	impl Serialize for Layout{
		fn serialize<S:Serializer>(&self,serializer:S)->StdResult<S::Ok,S::Error>{SerialLayout::Current(self.inner.clone()).serialize(serializer)}
	}

	#[derive(Deserialize,Serialize)]
	#[serde(untagged)]
	/// serial layout structure to ensure compatibility with legacy files
	enum SerialLayout{
		Current(Arc<(Vec<usize>,Vec<isize>)>),
		Old{dims:Vec<usize>,stray:bool,strides:Vec<isize>}
	}

	use serde::{Deserialize,Deserializer,Serialize,Serializer};
	use std::result::Result as StdResult;
	use super::*;
}

/*impl Validity{
	#[track_caller]
	/// construct the validity state of a layout
	pub fn try_new(dims:&[usize],strides:&[isize])->Result<Validity>{
		let mut count :usize=1;
		let mut length:usize=1;
		let rank=error::checked_rank(dims,strides).map_err(|e|e.with_op("new"))?;
																// check shared validity and length requirement
		for ix in 0..rank{
			let (dim,stride)=(dims[ix],strides[ix]);
			let c=count.checked_mul(dim);

			if dim>isize::MAX as usize{return Err(Error::  dim_overflow(error::diagnostic_shape(dims),"new"))}
			if c.is_none()            {return Err(Error::count_overflow(error::diagnostic_shape(dims),"new"))}

			length=length.saturating_add((dim-1).saturating_mul(stride.abs() as usize));
			if length>isize::MAX as usize{return Err(Error::len_overflow(error::diagnostic_layout(dims,strides),"new"))}

			count=c.unwrap();
		}
																// check mut validity
		/// compute the greatest common factor. helper funtion for lcm
		fn gcd(a:usize,b:usize)->usize{
			if a==b{return a}

			let (mut a,mut b)=(a.max(b),a.min(b));
			while b>0{(a,b)=(b,a%b)}

			a
		}
		/// compute the lowest common multiple. helper function for checking stride overlap
		fn lcm(a:usize,b:usize)->usize{a/gcd(a,b)*b}
																// trivial cases: empty and scalar can't alias
		if count==1||length==0{return Ok(Validity::Mut(length))}
																// check each axis
		for ix in 0..dims.len(){
																// unsqueeze case: if axis dim is 1, this dim may be skipped since unsqueezing doesn't alias. dim=1 -> only valid indices normalize to 0 -> element not accessible through multiple distinct indices along this axis
			if dims[ix]==1{continue}
																// broadcast case: when an axis overlaps with itself due to broadcasting, the stride is 0. This scenario guarantees alias when dim>1. Axes with dim <= 1 have already been handled above
			if strides[ix]==0{return Ok(Validity::Shared(length))}
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
					return Ok(Validity::Shared(length))
				}
			}
		}
		Ok(Validity::Mut(length))
	}
}
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
/// Describes the strongest validity guarantee satisfied by a layout together with the minimum backing buffer length it requires. Use in an Option or Result to describe the validity of possibly invalid layouts.
/// Layouts have 2 varying degrees of validity. A layout may have either no validity (invalid), shared validity (shared-valid), or mutable validity (mut-valid), with the difference between shared-valid and mut-valid being that mut-valid layouts are not allowed to have multiple positions refer to the same component. A layout is considered invalid if any of its dims exceed isize::MAX, if their product exceeds usize::MAX, if its buffer length overflows isize, or if its dims and strides have mismatched ranks. Layouts that are not invalid are considered 'valid', however, a valid layout is only 'valid for' buffer lengths greater than the greatest component offset of a tensor with that layout.
/// The default value is Mut(0). The variant is chosen to align with Tensor, which maintains mutable validity to allow disjoint mutable access to its components, and the numeric value is chosen to align with Tensor::default, which is an empty rank 1 tensor that fits in an empty buffer.
pub enum Validity{
	/// A layout is shared-valid for a len if every tensor position maps to a well-defined buffer offset less than that len
	Shared(usize),
	/// A layout is mut-valid for a len if every tensor position injectively maps to a well-defined buffer offset less than that len
	Mut(usize)
}*/

#[derive(Clone,Debug,Default,Eq,Hash,PartialEq)]
#[repr(transparent)]
/// Describes the shape and memory layout of a tensor.  A `Layout` stores the dimensions and strides of a tensor.
/// It does not contain an offset as owned tensors are eagerly trimmed and tensor views offset their pointers rather than their layouts.
/// Layouts have 2 varying degrees of validity. A layout may have either no validity (invalid), shared validity (shared-valid), or mutable validity (mut-valid), with the difference between shared-valid and mut-valid being that mut-valid layouts are not allowed to have multiple positions refer to the same component. A layout is considered invalid if any of its dims exceed isize::MAX, if their product exceeds usize::MAX, if its buffer length overflows isize, or if its dims and strides have mismatched ranks. Layouts that are not invalid are considered 'valid', however, a valid layout is only 'valid for' buffer lengths greater than the greatest component offset of a tensor with that layout.
pub struct Layout{inner:Arc<(Vec<usize>,Vec<isize>)>}

use std::{
	ops::{Bound,RangeBounds},sync::Arc
};
use super::{
	Error,Result,error,position::{PositionIter,SignedIndexPosition,self}
};
