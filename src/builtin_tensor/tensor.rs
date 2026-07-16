impl<E> AsRef<Self> for Tensor<E>{
	fn as_ref(&self)->&Self{self}
}
impl<E> AsRef<Self> for Tens  <E>{
	fn as_ref(&self)->&Self{self}
}
impl<E> AsMut<Self> for Tensor<E>{
	fn as_mut(&mut self)->&mut Self{self}
}
impl<E> AsMut<Self> for Tens  <E>{
	fn as_mut(&mut self)->&mut Self{self}
}
impl<E> AsRef    <View<E>> for Tens  <E>{
	fn as_ref(&    self)->&    View<E>{self  .as_view()}
}
impl<E> AsRef    <View<E>> for Tensor<E>{
	fn as_ref(&    self)->&    View<E>{self.0.as_view()}
}
impl<E> AsMut    <View<E>> for Tens  <E>{
	fn as_mut(&mut self)->&mut View<E>{self  .as_mut_view()}
}
impl<E> AsMut    <View<E>> for Tensor<E>{
	fn as_mut(&mut self)->&mut View<E>{self.0.as_mut_view()}
}
impl<E> Borrow   <View<E>> for Tens  <E>{
	fn borrow(&    self)->&    View<E>{self  .as_view()}
}
impl<E> Borrow   <View<E>> for Tensor<E>{
	fn borrow(&    self)->&    View<E>{self.0.as_view()}
}
impl<E> BorrowMut<View<E>> for Tens  <E>{
	fn borrow_mut(&mut self)->&mut View<E>{self  .as_mut_view()}
}
impl<E> BorrowMut<View<E>> for Tensor<E>{
	fn borrow_mut(&mut self)->&mut View<E>{self.0.as_mut_view()}
}
impl<E:Clone> Clone for Tensor<E>{
	fn clone(&self)->Self{Self(self.0.clone())}
	fn clone_from(&mut self,other:&Self){self.0.clone_from(&other.0)}
}
impl<E:Clone> Clone for Tens  <E>{
	fn clone(&self)->Self{
		let buffer=self.buffer().to_vec();
		let layout=self.layout().clone();

		Self::from_inner(buffer,layout)
	}
	fn clone_from(&mut self,other:&Self){
		let mut buffer=self.take_buffer();

		buffer.clear();
		buffer.extend_from_slice(other.buffer());

		self.layout_mut().clone_from(other.layout());
		self.set_buffer(buffer);
	}
}
impl<E:Debug> Debug for Tensor<E>{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		#[allow(unused)]
		#[derive(Debug)]
		struct Tensor<'a,E:'a>{data:&'a [E],dims:&'a [usize]}
		Tensor{data:self.buffer(),dims:self.dims()}.fmt(f)
	}
}
impl<E:Debug> Debug for Tens<E>{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		#[allow(unused)]
		#[derive(Debug)]
		struct Tens<'a,E:'a>{buffer:&'a [E],dims:&'a [usize],strides:&'a [isize]}
		Tens{buffer:self.buffer(),dims:self.dims(),strides:self.strides()}.fmt(f)
	}
}
impl<E> Default  for Tensor<E>{
	fn default()->Self{Self::empty(1)}
}
impl<E> Default  for Tens  <E>{
	fn default()->Self{Self::empty(1)}
}
impl<E> Deref    for Tens  <E>{
	fn deref    (&    self)->&    View<E>{self  .as_view()}
	type Target=View<E>;
}
impl<E> Deref    for Tensor<E>{
	fn deref    (&    self)->&    View<E>{self.0.as_view()}
	type Target=View<E>;
}
impl<E> DerefMut for Tens  <E>{
	fn deref_mut(&mut self)->&mut View<E>{self  .as_mut_view()}
}
impl<E> DerefMut for Tensor<E>{
	fn deref_mut(&mut self)->&mut View<E>{self.0.as_mut_view()}
}
impl<E> Drop for Tens<E>{
	fn drop(&mut self){
		let (ptr,len,cap)=(self.ptr,self.len,self.cap);
		unsafe{			// for safety, a valid buffer is maintained when cap>0. If cap==0, this is a field of a borrowed view and should not be dropped.
			if cap==0{return}
			mem::drop(Vec::from_raw_parts(ptr,len,cap));
		}
	}
}
impl<E:Eq> Eq for Tensor<E>{}
impl<E:Eq> Eq for Tens  <E>{}
impl<E> FromIterator<E> for Tensor<E>{
	fn from_iter<I:IntoIterator<Item=E>>(iter:I)->Self{Vec::from_iter(iter).into()}
}
impl<E> FromIterator<E> for Tens  <E>{
	fn from_iter<I:IntoIterator<Item=E>>(iter:I)->Self{Vec::from_iter(iter).into()}
}
/*// can't do this unfortunately due to ty
impl<E> From<E> for Tensor<E>{
	fn from(data:E)->Self{Self::scalar(data)}
}
impl<E> From<E> for Tens  <E>{
	fn from(data:E)->Self{Self::scalar(data)}
}*/
impl<E> From<Tensor<E>> for Tens<E>{
	fn from(data:Tensor<E>)->Self{data.0}
}
impl<E:Clone> From<&View<E>> for Tens<E>{
	fn from(data:&View<E>)->Self{data.to_tens()}
}
impl<E:Clone> From<&mut View<E>> for Tens<E>{
	fn from(data:&mut View<E>)->Self{data.to_tens()}
}
impl<'a,E:Clone> From<ViewRef<'a,E>> for Tensor<E>{
	fn from(data:ViewRef<'a,E>)->Self{
		let buffer=data.flat_vec(None);
		let layout=Layout::new(data.dims());

		Self(Tens::from_inner(buffer,layout))
	}
}
impl<'a,E:Clone> From<ViewMut<'a,E>> for Tensor<E>{
	fn from(data:ViewMut<'a,E>)->Self{
		let buffer=data.flat_vec(None);
		let layout=Layout::new(data.dims());

		Self(Tens::from_inner(buffer,layout))
	}
}
impl<E> From<Vec<E>> for Tensor<E>{
	fn from(data:Vec<E>)->Self{Self::vector(data)}
}
impl<E> From<Vec<E>> for Tens  <E>{
	fn from(data:Vec<E>)->Self{Self::vector(data)}
}
impl<E:Hash> Hash for Tensor<E>{
	fn hash<H:Hasher>(&self,state:&mut H){(**self).hash(state)}
}
impl<E:Hash> Hash for Tens  <E>{
	fn hash<H:Hasher>(&self,state:&mut H){(**self).hash(state)}
}
impl<E,P:SignedIndexPosition,const N:usize> Index<[P;N]> for Tensor<E>{
	#[track_caller]
	fn index(&self,index:[P;N])->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E,P:SignedIndexPosition,const N:usize> Index<[P;N]> for Tens  <E>{
	#[track_caller]
	fn index(&self,index:[P;N])->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E,P:SignedIndexPosition> Index<&[P]> for Tensor<E>{
	#[track_caller]
	fn index(&self,index:&[P])->&Self::Output{self.as_view().index(index)}
	type Output=E;
}
impl<E,P:SignedIndexPosition> Index<&[P]> for Tens  <E>{
	#[track_caller]
	fn index(&self,index:&[P])->&Self::Output{self.as_view().index(index)}
	type Output=E;
}
impl<E> Index< Position> for Tensor<E>{
	#[track_caller]
	fn index(&self,index:Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E> Index< Position> for Tens  <E>{
	#[track_caller]
	fn index(&self,index:Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E> Index<&Position> for Tensor<E>{
	#[track_caller]
	fn index(&self,index:&Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E> Index<&Position> for Tens  <E>{
	#[track_caller]
	fn index(&self,index:&Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E,P:SignedIndexPosition,const N:usize> IndexMut<[P;N]> for Tensor<E>{
	#[track_caller]
	fn index_mut(&mut self,index:[P;N])->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E,P:SignedIndexPosition,const N:usize> IndexMut<[P;N]> for Tens  <E>{
	#[track_caller]
	fn index_mut(&mut self,index:[P;N])->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E,P:SignedIndexPosition> IndexMut<&[P]> for Tensor<E>{
	#[track_caller]
	fn index_mut(&mut self,index:&[P])->&mut Self::Output{self.as_mut_view().index_mut(index)}
}
impl<E,P:SignedIndexPosition> IndexMut<&[P]> for Tens  <E>{
	#[track_caller]
	fn index_mut(&mut self,index:&[P])->&mut Self::Output{self.as_mut_view().index_mut(index)}
}
impl<E> IndexMut< Position> for Tensor<E>{
	#[track_caller]
	fn index_mut(&mut self,index:Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E> IndexMut< Position> for Tens  <E>{
	#[track_caller]
	fn index_mut(&mut self,index:Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E> IndexMut<&Position> for Tensor<E>{
	#[track_caller]
	fn index_mut(&mut self,index:&Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E> IndexMut<&Position> for Tens  <E>{
	#[track_caller]
	fn index_mut(&mut self,index:&Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<   E:PartialEq<X>,X> PartialEq<Tensor <   X>> for Tensor<E>{
	fn eq(&self,other:&Tensor <   X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&Tensor <   X>)->bool{(**self)!=(**other)}
}
impl<   E:PartialEq<X>,X> PartialEq<Tensor <   X>> for Tens  <E>{
	fn eq(&self,other:&Tensor <   X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&Tensor <   X>)->bool{(**self)!=(**other)}
}
impl<   E:PartialEq<X>,X> PartialEq<Tens   <   X>> for Tensor<E>{
	fn eq(&self,other:&Tens   <   X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&Tens   <   X>)->bool{(**self)!=(**other)}
}
impl<   E:PartialEq<X>,X> PartialEq<Tens   <   X>> for Tens  <E>{
	fn eq(&self,other:&Tens   <   X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&Tens   <   X>)->bool{(**self)!=(**other)}
}
impl<'b,E:PartialEq<X>,X> PartialEq<ViewRef<'b,X>> for Tensor<E>{
	fn eq(&self,other:&ViewRef<'b,X>)->bool{(**self)==(*other)}
	fn ne(&self,other:&ViewRef<'b,X>)->bool{(**self)!=(*other)}
}
impl<'b,E:PartialEq<X>,X> PartialEq<ViewRef<'b,X>> for Tens  <E>{
	fn eq(&self,other:&ViewRef<'b,X>)->bool{(**self)==(*other)}
	fn ne(&self,other:&ViewRef<'b,X>)->bool{(**self)!=(*other)}
}
impl<'b,E:PartialEq<X>,X> PartialEq<ViewMut<'b,X>> for Tensor<E>{
	fn eq(&self,other:&ViewMut<'b,X>)->bool{(**self)==(*other)}
	fn ne(&self,other:&ViewMut<'b,X>)->bool{(**self)!=(*other)}
}
impl<'b,E:PartialEq<X>,X> PartialEq<ViewMut<'b,X>> for Tens  <E>{
	fn eq(&self,other:&ViewMut<'b,X>)->bool{(**self)==(*other)}
	fn ne(&self,other:&ViewMut<'b,X>)->bool{(**self)!=(*other)}
}
impl<   E:PartialEq<X>,X> PartialEq<View   <   X>> for Tensor<E>{
	fn eq(&self,other:&View   <   X>)->bool{(**self)==(*other)}
	fn ne(&self,other:&View   <   X>)->bool{(**self)!=(*other)}
}
impl<   E:PartialEq<X>,X> PartialEq<View   <   X>> for Tens  <E>{
	fn eq(&self,other:&View   <   X>)->bool{(**self)==(*other)}
	fn ne(&self,other:&View   <   X>)->bool{(**self)!=(*other)}
}
impl<E> TryFrom<Tens<E>> for Tensor<E>{
	fn try_from(inner:Tens<E>)->Result<Self>{
		inner.validate_mut().map_err(|e|e.with_op("tensor"))?;
		Ok(Self(inner))
	}
	type Error=Error;
}
impl<E:Clone> TryFrom<&View<E>> for Tensor<E>{
	fn try_from(inner:&View<E>)->Result<Self>{
		inner.validate_mut().map_err(|e|e.with_op("tensor"))?;
		Ok(Self(inner.to_tens()))
	}
	type Error=Error;
}
impl<E:Clone> TryFrom<&mut View<E>> for Tensor<E>{
	fn try_from(inner:&mut View<E>)->Result<Self>{
		inner.validate_mut().map_err(|e|e.with_op("tensor"))?;
		Ok(Self(inner.to_tens()))
	}
	type Error=Error;
}

impl<E> Tensor<E>{
	#[track_caller]
	/// broadcast a specific axis. panics if the layout is invalid, if the dims would be incompatible, or if dims and strides have mismatched lengths. panics if the index is out of bounds
	pub fn broadcast_dim(mut self,index:impl SignedIndexPosition,rhs:usize)->Self where E:Clone{
		self.0.broadcast_dim(index,rhs);
		self
	}
	#[track_caller]
	/// try broadcasting the dims, panicing if the dims are not broadcast compatible with rh
	pub fn broadcast<D:AsRef<[usize]>>(&mut self,rhs:D)->&mut Self where E:Clone{
		self.0.broadcast(rhs);
		self
	}
	/// get the buffer capacity
	pub fn buffer_cap(&self)->usize{self.0.cap}
	/// get the buffer len
	pub fn buffer_len(&self)->usize{self.0.len}
	/// reference the buffer
	pub fn buffer(&self)->&[E]{self.0.buffer()}
	/// reference the buffer
	pub fn buffer_mut(&mut self)->&mut [E]{self.0.buffer_mut()}
	#[track_caller]
	/// create an empty tensor with the specified rank. panics if rank is 0
	pub fn empty(rank:usize)->Self{Self(Tens::empty(rank))}
	#[track_caller]
	/// reverse the order of components along all axes except the one at the index. panics if the index is out of bounds
	pub fn flip_around(mut self,index:impl SignedIndexPosition)->Self{
		self.0.flip_around(index);
		self
	}
	#[track_caller]
	/// reverse the order of components along the axis. panics if the index is out of bounds
	pub fn flip_dim(mut self,index:impl SignedIndexPosition)->Self{
		self.0.flip_dim(index);
		self
	}
	/// reverse the order of components along all axes
	pub fn flip(mut self)->Self{
		self.0.flip();
		self
	}
	/// unwrap the inner buffer
	pub fn into_buffer(self)->Vec<E>{self.0.into_buffer()}
	/// convert into the inner data
	pub fn into_inner(self)->(Vec<E>,Layout){self.0.into_inner()}
	/// convert into an owned view
	pub fn into_unique<'a>(self)->ViewRef<'a,E> where E:'a{
		let (buffer,layout)=self.into_inner();
		view::unique(buffer,layout)
	}
	/// convert into an owned view
	pub fn into_unique_mut<'a>(self)->ViewMut<'a,E> where E:'a{
		let (buffer,layout)=self.into_inner();
		view::unique_mut(buffer,layout)
	}
	/// apply a function to every component, returning a new tensor
	pub fn map<F:FnMut(E)->Y,Y>(mut self,f:F)->Tens<Y>{
		error::unwrap_or_panic(self.0.checked_normalize_layout().map_err(|e|e.with_op("map")));
											// with the layout normalized, this has predictable iteration order
		let (buffer,layout)=self.0.into_inner();
		Tens::from_inner(buffer.into_iter().map(f).collect(),layout)
	}
	#[track_caller]
	/// create a new tensor. Err if the dims have a product greater than the data len
	pub fn new(data:Vec<E>,dims:impl AsRef<[usize]>)->Self{error::unwrap_or_panic(Self::try_new(data,dims))}
	/// create a 0d tensor from a scalar
	pub fn scalar(data:E)->Self{Self(Tens::scalar(data))}
	/// set the layout. panic if the layout is invalid for the buffer
	pub fn set_layout(&mut self,layout:Layout){error::unwrap_or_panic(self.try_set_layout(layout))}
	#[track_caller]
	/// slice dim. panics if the index or range are out of bounds
	pub fn slice_dim<I:SignedIndexPosition>(mut self,index:impl SignedIndexPosition,range:impl RangeBounds<I>)->Self{
		self.0.slice_dim(index,range);
		self
	}
	#[track_caller]
	/// slice. panics if the range are out of bounds
	pub fn slice<I:SignedIndexPosition,R:RangeBounds<I>>(mut self,ranges:&[R])->Self{
		self.0.slice(ranges);
		self
	}
	#[track_caller]
	/// squeeze an axis of dim 1 into nonexistence. panics if the dim at the index is not equal to 1. panics if out of bounds of the rank
	pub fn squeeze_dim(mut self,index:impl SignedIndexPosition)->Self{
		self.0.squeeze_dim(index);
		self
	}
	#[track_caller]
	/// swap a pair of axes.
	pub fn swap_dims(mut self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->Self{
		self.0.swap_dims(a,b);
		self
	}
	/// create a new tensor. Err if the dims have a product greater than the data len
	pub fn try_new(data:Vec<E>,dims:impl AsRef<[usize]>)->Result<Self>{Tens::try_new(data,dims).map(Tens::tensor)}
	/// set the layout. Err if the layout is invalid for the buffer
	pub fn try_set_layout(&mut self,layout:Layout)->Result<()>{
		layout.validate_mut(self.buffer_len()).map_err(|e|e.with_op("reshape"))?;
		Ok(self.0.layout=layout)
	}
	/// convert back into a Tens
	pub fn tens(self)->Tens<E>{self.0}
	#[track_caller]
	/// unsqueeze an axis of dim 1 into existence. panics if out of bounds of the rank
	pub fn unsqueeze_dim(mut self,index:impl SignedIndexPosition)->Self{
		self.0.unsqueeze_dim(index);
		self
	}
	/// create a 1d tensor from a vector
	pub fn vector(data:Vec<E>)->Self{Self(Tens::vector(data))}
	/// set the layout. panic if the layout is invalid for the buffer
	pub fn with_layout(mut self,layout:Layout)->Self{
		self.set_layout(layout);
		self
	}
}
impl<E> Tens<E>{
	/// get the pointer to the buffer
	pub fn as_mut_ptr(&mut self)->*mut E{self.ptr}// break deref cycle
	/// get the pointer to the buffer
	pub fn as_ptr(&self)->*const E{self.ptr as *const E}// break deref cycle
	/// reference as a View
	pub fn as_view(&self)->&View<E>{// break deref cycle
		unsafe{		// safety: View is a transparent of [Self]. validity is only guaranteed for borrowed view
			mem::transmute(slice::from_ref(self))
		}
	}
	/// reference as a View
	pub fn as_mut_view(&mut self)->&mut View<E>{// break deref cycle
		unsafe{		// safety: View is a transparent of [Self]. validity is only guaranteed for borrowed view
			mem::transmute(slice::from_mut(self))
		}
	}
	#[track_caller]
	/// broadcast a specific axis. panics if the layout is invalid, or if the dims would be incompatible. panics if the index is out of bounds
	pub fn broadcast_dim(&mut self,index:impl SignedIndexPosition,rhs:usize)->&mut Self where E:Clone{
		*self=self.view().broadcast_dim(index,rhs).to_tens();
		self
	}
	#[track_caller]
	/// try broadcasting the dims. panics if the layout is invalid, or if the dims would be incompatible.
	pub fn broadcast<D:AsRef<[usize]>>(&mut self,rhs:D)->&mut Self where E:Clone{
		*self=self.view().broadcast(rhs).to_tens();
		self
	}
	/// get the buffer capacity
	pub fn buffer_cap(&self)->usize{self.cap}
	/// get the buffer len
	pub fn buffer_len(&self)->usize{self.len}
	/// reference the buffer
	pub fn buffer_mut(&mut self)->&mut [E]{
		unsafe{		// safety: slice::from_raw_parts_mut is weaker than the postcondition of Self::_from_raw_parts: When (ptr, len, cap) are not ok to put in Vec::from_raw_parts (borrowed buffer case), the resulting Tens must never convert its buffer to a slice or vec. Ensure all construction goes through _from_raw_parts.
			slice::from_raw_parts_mut(self.as_mut_ptr(),self.buffer_len())
		}
	}
	/// reference the buffer
	pub fn buffer(&self)->&[E]{
		unsafe{		// safety: slice::from_raw_parts is weaker than the postcondition of Self::_from_raw_parts: When (ptr, len, cap) are not ok to put in Vec::from_raw_parts (borrowed buffer case), the resulting Tens must never convert its buffer to a slice or vec. Ensure all construction goes through _from_raw_parts.
			slice::from_raw_parts(self.as_ptr(),self.buffer_len())
		}
	}
	/// normalize the tensor to a contiguous layout. returns Err if the layout is not mutably valid for the buffer
	pub fn checked_normalize_layout(&mut self)->Result<&mut Self>{
		self.validate_mut().map_err(|e|e.with_op("rearrange"))?;
		Ok(self.normalize_layout())
	}
	/// reference the dims. Note that producing a Tens with a layout invalid for its buffer is allowed, but may lead to incorrect behavior or panics on functions that assume layout validity
	pub fn dims_mut(&mut self)->&mut [usize]{self.layout_mut().dims_mut()}
	#[track_caller]
	/// create an empty tensor with the specified rank. rank=0 is not allowed because it would need to have 1 component
	pub fn empty(rank:usize)->Self{
		assert!(rank>0);

		let buffer=Vec::new();
		let layout=Layout::new(vec![0;rank]);

		Self::from_inner(buffer,layout)
	}
	#[track_caller]
	/// reverse the order of components along all axes except the one at the index. panics if the index is out of bounds
	pub fn flip_around(&mut self,index:impl SignedIndexPosition)->&mut Self{
		self.layout.flip_around(index);
		self
	}
	#[track_caller]
	/// reverse the order of components along the axis. panics if the index is out of bounds
	pub fn flip_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		self.layout.flip_dim(index);
		self
	}
	/// reverse the order of components along all axes
	pub fn flip(&mut self)->&mut Self{
		self.layout.flip();
		self
	}
	/// create a Tens from its inner data without validating
	pub fn from_inner(mut buffer:Vec<E>,layout:Layout)->Self{
		unsafe{		// safety: (ptr, len, cap) are converted from Vec as in the official documentation example
			let ptr=buffer.as_mut_ptr();
			let len=buffer.len();
			let cap=buffer.capacity();

			mem::forget(buffer);
			Self::from_raw_parts(layout,ptr,len,cap)
		}
	}
	/// create a Tens from raw parts. If cap>0, (ptr, len, cap) must be ok to convert to Vec. Even if cap==0, valid layouts (in general not just for buffer) stored by the resulting Tens must never produce an offset less than len for which ptr+offset not valid for conversion to a shared reference, and if the buffer will be mutated, temporary conversion to a mutable reference. When (ptr, len, cap) are not ok to put in Vec::from_raw_parts (borrowed buffer case), the resulting Tens must never convert its buffer to a slice or vec.
	pub (crate) unsafe fn _from_raw_parts(layout:Layout,ptr:*mut E,len:usize,cap:usize)->Self{
		Self{layout,ptr,len,cap}
	}
	/// create a Tens from raw parts. (ptr, len, cap) must be ok to convert to Vec.
	pub unsafe fn from_raw_parts(layout:Layout,ptr:*mut E,len:usize,cap:usize)->Self{
		unsafe{		// safety: the public precondition actually is stronger than the private one as it implies len<=cap, which trivially satisfies the cap==0 case. The public precondition also forbids the borrowed buffer case.
			Self::_from_raw_parts(layout,ptr,len,cap)
		}
	}
	/// unwrap the inner buffer
	pub fn into_buffer(self)->Vec<E>{self.into_inner().0}
	/// flatten into a vec. may have unexpected results if the layout is not mutably valid
	pub fn into_flat_vec(mut self)->Vec<E>{
		self.normalize_layout();
		self.into_buffer()
	}
	/// convert into the inner data
	pub fn into_inner(mut self)->(Vec<E>,Layout){
		unsafe{		// safety: postcondition of Self::_from_raw_parts: When (ptr, len, cap) are not ok to put in Vec::from_raw_parts (borrowed buffer case), the resulting Tens must never convert its buffer to a slice or vec. Ensure all construction goes through _from_raw_parts.
			let buffer=Vec::from_raw_parts(self.ptr,self.len,self.cap);
			let layout=self.layout.clone();
					// unfortunately, mem::forget would memory leak the layout, so we instead set cap to 0 to make the Tens::drop implementation not drop the buffer
			self.cap=0;
			(buffer,layout)
		}
	}
	#[track_caller]
	/// convert into an owned view. panics if the layout is invalid
	pub fn into_unique_ref<'a>(self)->ViewRef<'a,E> where E:'a{
		let (buffer,layout)=self.into_inner();
		view::unique(buffer,layout)
	}
	#[track_caller]
	/// convert into an owned view. panics if the layout is invalid
	pub fn into_unique_mut<'a>(self)->ViewMut<'a,E> where E:'a{
		let (buffer,layout)=self.into_inner();
		view::unique_mut(buffer,layout)
	}
	/// reference the layout
	pub fn layout(&self)->&Layout{&self.layout}
	/// reference the layout
	pub fn layout_mut(&mut self)->&mut Layout{&mut self.layout}
	#[track_caller]
	/// apply a function to every component. panics if the layout is not mutably valid for the buffer
	pub fn map<F:FnMut(&mut E)>(&mut self,mut f:F)->&mut Self{
		error::unwrap_or_panic(self.validate_mut().map_err(|e|e.with_op("map")));

		if self.is_layout_normalized(){
			self.buffer_mut().iter_mut().for_each(f)
		}else{
			for px in self.positions(){f(&mut self[px])}
		};
		self
	}
	#[track_caller]
	/// create a new tensor. Err if the dims have a product greater than the data len
	pub fn new(data:Vec<E>,dims:impl AsRef<[usize]>)->Self{error::unwrap_or_panic(Self::try_new(data,dims))}
	/// normalize the tensor to a contiguous layout. may have unexpected results if the layout is not mutably valid, but will always result in a tensor for which is_layout_normalized returns true
	pub fn normalize_layout(&mut self)->&mut Self{
		if self.is_layout_normalized(){return self}
											// return early if already normalized or if can't be normalized. We can convert an invalid layout to a normalized layout for this buffer by making into a rank 1 tensor
		if self.validate_mut().is_err(){
			self.layout=Layout::new([self.len]);// TODO should this be a specified behavior
			return self;
		}
		let newlayout=Layout::new(self.dims());

		let (mut buffer,layout)=mem::replace(self,Self::from_inner(Vec::new(),newlayout)).into_inner();
		let mut temp:Vec<Option<E>>=buffer.drain(..).map(Some).collect();

		for px in layout.positions(){buffer.push(temp[layout.compute_offset(&px)].take().unwrap())}

		self.set_buffer(buffer);
		self
	}
	/// swap out the buffer.  Note that producing a Tens with a layout not mutably valid for its buffer is allowed, but may lead to incorrect behavior or panics on functions that assume layout validity
	pub fn replace_buffer(&mut self,mut replacement:Vec<E>)->Vec<E>{
		unsafe{		// safety: postcondition of Self::_from_raw_parts: When (ptr, len, cap) are not ok to put in Vec::from_raw_parts (borrowed buffer case), the resulting Tens must never convert its buffer to a slice or vec. Ensure all construction goes through _from_raw_parts.
			let old=Vec::from_raw_parts(mem::replace(&mut self.ptr,replacement.as_mut_ptr()),mem::replace(&mut self.len,replacement.len()),mem::replace(&mut self.cap,replacement.len()));
			mem::forget(replacement);

			old
		}
	}
	/// create a 0d tensor from a scalar
	pub fn scalar(data:E)->Self{
		Self::from_inner(vec![data],Layout::scalar())
	}
	/// set the buffer. Note that producing a Tens with a not mutably valid for its buffer is allowed, but may lead to incorrect behavior or panics on functions that assume layout validity
	pub fn set_buffer(&mut self,buffer:Vec<E>){
		self.replace_buffer(buffer);
	}
	#[track_caller]
	/// slice dim. panics if the index or range are out of bounds or if the layout is invalid for the buffer. does not require mut layout validity, only shared
	pub fn slice_dim<I:SignedIndexPosition>(&mut self,index:impl SignedIndexPosition,range:impl RangeBounds<I>)->&mut Self{
		error::unwrap_or_panic(self.validate());

		let mut buffer=self.take_buffer();
		let mut offset=0;

		self.layout.slice_dim(index,&mut offset,range);

		buffer.drain(..offset);
		buffer.truncate(self.layout.len());

		self.set_buffer(buffer);
		self
	}
	#[track_caller]
	/// slice. panics if the range are out of bounds or if the layout is invalid for the buffer. does not require mut layout validity, only shared
	pub fn slice<I:SignedIndexPosition,R:RangeBounds<I>>(&mut self,ranges:&[R])->&mut Self{
		error::unwrap_or_panic(self.validate());

		let mut buffer=self.take_buffer();
		let mut offset=0;

		self.layout.slice(&mut offset,ranges);

		buffer.drain(..offset);
		buffer.truncate(self.layout.len());

		self.set_buffer(buffer);
		self
	}
	#[track_caller]
	/// split in two at the position. panics if out of bounds or the layout is invalid for the buffer
	pub fn split_off(&mut self,index:impl SignedIndexPosition,position:impl SignedIndexPosition)->Self{
		error::unwrap_or_panic(self.validate());
		self.swap_dims(0,index).normalize_layout();

		let dim =self.dims()[0];
		let position=if let Some(px)=position::unsign_position(dim,position){px}else{panic!("position {} is out of bounds for dim {dim}",position.expect_isize("must be able to convert index to isize"))};

		let at=position::compute_offset(&[dim],&[position],&[self.strides()[0]]);
		let rldim=position;
		let rllayout=self.layout_mut();
		let rrdim=dim-position;
		let mut rrlayout=rllayout.clone();

		(rllayout.dims_mut()[0],rrlayout.dims_mut()[0])=(rldim,rrdim);

		let mut buffer=self.take_buffer();
		let newbuffer =buffer.split_off(at);

		self.set_buffer(buffer);
		self.swap_dims(0,index);

		let mut result=Tens::from_inner(newbuffer,rrlayout);
		result.swap_dims(0,index);

		result
	}
	#[track_caller]
	/// squeeze an axis of dim 1 into nonexistence. panics if the dim at the index is not equal to 1. panics if out of bounds of the rank. does not check buffer validity as this only modifies the layout
	pub fn squeeze_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		self.layout.squeeze_dim(index);
		self
	}
	/// reference the dims. Note that producing a Tens with a layout invalid for its buffer is allowed, but may lead to incorrect behavior or panics on functions that assume layout validity
	pub fn strides_mut(&mut self)->&mut [isize]{self.layout_mut().strides_mut()}
	#[track_caller]
	/// swap a pair of axes. unspecified result if the layout is invalid for the buffer
	pub fn swap_dims(&mut self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->&mut Self{
		self.layout.swap_dims(a,b);
		self
	}
	/// take the buffer, replacing it with an empty buffer
	pub fn take_buffer(&mut self)->Vec<E>{
		self.replace_buffer(Vec::new())
	}
	/// convert to a tensor value that operates in a functional style. panic if the layout is invalid for the buffer
	pub fn tensor(self)->Tensor<E>{
		error::unwrap_or_panic(self.layout().validate_mut(self.buffer_len()).map_err(|e|e.with_op("tensor")));
		Tensor(self)
	}
	/// create a new tensor. Err if the dims have a product greater than the data len
	pub fn try_new(data:Vec<E>,dims:impl AsRef<[usize]>)->Result<Self>{
		let layout=Layout::try_new(dims)?;
		layout.validate(data.len()).map_err(|e|e.with_op("new"))?;

		Ok(Self::from_inner(data,layout))
	}
	#[track_caller]
	/// squeeze an axis of dim 1 into nonexistence. panics if out of bounds of the rank. does not check buffer validity as this only modifies the layout
	pub fn unsqueeze_dim(&mut self,index:impl SignedIndexPosition)->&mut Self{
		self.layout.unsqueeze_dim(index);
		self
	}
	/// create a 1d tensor from a vector
	pub fn vector(data:Vec<E>)->Self{
		let layout=Layout::new([data.len()]);
		Self::from_inner(data,layout)
	}
}

#[cfg(feature="serial")]
mod serial{
	impl<'a,E:Deserialize<'a>> Deserialize<'a> for Tens<E>{
		fn deserialize<D:Deserializer<'a>>(deserializer:D)->StdResult<Self,D::Error>{
			#[derive(Deserialize)]
			struct Data<E>{data:Vec<E>,layout:Layout}
			let t=Data::deserialize(deserializer)?;

			Ok(Tens::from_inner(t.data,t.layout))
		}
	}
	impl<E:Serialize> Serialize for Tens<E>{
		fn serialize<S:Serializer>(&self,serializer:S)->StdResult<S::Ok,S::Error>{
			#[derive(Serialize)]
			struct Data<'a,E:'a>{data:&'a [E],layout:Layout}
			Data{data:self.buffer(),layout:self.get_layout()}.serialize(serializer)
		}
	}

	use serde::{Deserialize,Deserializer,Serialize,Serializer};
	use std::{result::Result as StdResult};
	use super::{Layout,Tens};
}
#[cfg(test)]
mod tests{
	#[test]
	fn index_vector(){
		let tensor=Tensor::vector(vec![1,2,3,4,5]);
		let values=vec![1,2,3,4,5];

		let v:Vec<i32>=(0..5).map(|px|tensor[[px]]).collect();

		assert_eq!(values,v);
	}
	#[test]
	fn map(){
		let mut tens=Tens::vector(vec![101,102, 201,202, 301,302,
	                                   111,112, 211,212, 311,312]);
		let mut ten2=Tens::vector(vec![102,103, 202,203, 302,303,
		                               112,113, 212,213, 312,313]);
		tens.layout=Layout::from_inner(vec![3,2,2],vec![2,6,1]);
		ten2.layout=Layout::from_inner(vec![3,2,2],vec![2,6,1]);

		let tensor=tens.clone().tensor().map(|x|x+1);
		tens.map(|x|*x+=1);

		assert_eq!(tens,ten2);
		assert_eq!(tensor,ten2);
	}
	#[test]
	fn normalize_layout(){
		let mut tens=Tens::vector(vec![101,102, 201,202, 301,302,
		                               111,112, 211,212, 311,312]);
		tens.layout=Layout::from_inner(vec![3,2,2],vec![2,6,1]);

		let mut normalized=tens.clone();

		assert!(!normalized.is_layout_normalized());
		normalized.normalize_layout();
		assert!( normalized.is_layout_normalized());

		assert_eq!(normalized,tens);
		assert_eq!(normalized.buffer() ,[101,102,111,112,201,202,211,212,301,302,311,312]);
		assert_eq!(normalized.dims()   ,[3,2,2]);
		assert_eq!(normalized.strides(),[4,2,1]);
	}
	#[test]
	fn split_off(){
		let mut tens=Tens::vector(vec![101,102, 201,202, 301,302,
		                               111,112, 211,212, 311,312]);
		tens.layout=Layout::from_inner(vec![3,2,2],vec![2,6,1]);

		let ten2=tens.split_off(0,1);

		assert_eq!(tens,Tens::new(vec![101,102, 111,112],[1,2,2]));
		assert_eq!(ten2,Tens::new(vec![201,202, 211,212,
		                               301,302, 311,312],[2,2,2]));
	}


	use super::*;
}

/// A growable multidimensional array type, abbreviated from Tensor (Similar to Vec). Layout validity is lazily checked, and producing a Tens with a layout invalid for its buffer is allowed, but failing to maintain mutable validity may lead to panics or unexpected behavior of some functions. The default value is an empty tensor with dims [0]
pub struct Tens<E>{
	layout:Layout,	// layout info, possibly modified
	ptr:*mut E,		// buffer pointer.
	len:usize,		// buffer length
	cap:usize,		// buffer capacity. 0 if not owned (when this is used as a View field). If nonzero, (ptr, len, cap) must be ok to convert to Vec
}
#[repr(transparent)]
/// A tensor value that operates in a functional style, with mostly chain-move functions rather than chain-mut functions. mutable Layout validity is checked on construction. The default value is an empty tensor with dims [0]
pub struct Tensor<E>(Tens<E>);

use std::{
	borrow::{Borrow,BorrowMut},cmp::{Eq,PartialEq},fmt::{Debug,Formatter,Result as FmtResult},hash::{Hash,Hasher},iter::FromIterator,mem,ops::{Deref,DerefMut,Index,IndexMut,RangeBounds},slice
};
use super::{
	Error,Layout,Position,Result,View,error,position::{SignedIndexPosition,self},view::{ViewRef,ViewMut,self}
};
