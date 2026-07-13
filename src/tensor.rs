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
impl<E> From<E> for Tensor<E>{
	fn from(data:E)->Self{Self::scalar(data)}
}
impl<E> From<E> for Tens  <E>{
	fn from(data:E)->Self{Self::scalar(data)}
}
impl<E> From<Tensor<E>> for Tens<E>{
	fn from(data:Tensor<E>)->Self{data.0}
}
impl<E:Clone> From<&View<E>> for Tens<E>{
	fn from(data:&View<E>)->Self{data.to_tens()}
}
impl<E:Clone> From<&mut View<E>> for Tens<E>{
	fn from(data:&mut View<E>)->Self{data.to_tens()}
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
	/// get the buffer capacity
	pub fn buffer_cap(&self)->usize{self.0.cap}
	/// get the buffer len
	pub fn buffer_len(&self)->usize{self.0.len}
	/// reference the buffer
	pub fn buffer(&self)->&[E]{self.0.buffer()}
	/// reference the buffer
	pub fn buffer_mut(&mut self)->&mut [E]{self.0.buffer_mut()}
	/// create an empty tensor with the specified rank. panics if rank is 0
	pub fn empty(rank:usize)->Self{
		assert!(rank!=0);
		Self(Tens::empty(rank))
	}
	/// convert into the inner data
	pub fn into_inner(self)->(Vec<E>,Layout){self.0.into_inner()}
	#[track_caller]
	/// create a new tensor. Err if the dims have a product greater than the data len
	pub fn new(data:Vec<E>,dims:impl AsRef<[usize]>)->Self{error::unwrap_or_panic(Self::try_new(data,dims))}
	/// create a 0d tensor from a scalar
	pub fn scalar(data:E)->Self{Self(Tens::scalar(data))}
	/// set the layout. panic if the layout is invalid for the buffer
	pub fn set_layout(&mut self,layout:Layout){error::unwrap_or_panic(self.try_set_layout(layout))}
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
	/// swap a pair of axes.
	pub fn swap_dims(mut self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->Self{
		self.0.swap_dims(a,b);
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
	/// reference the dims. Note that producing a Tens with a layout invalid for its buffer is allowed, but may lead to incorrect behavior or panics on functions that assume layout validity
	pub fn dims_mut(&mut self)->&mut [usize]{self.layout_mut().dims_mut()}
	/// create an empty tensor with the specified rank. note that if rank is 0, the result will be invalid due to expecting a buffer of length 1 to hold a scalar
	pub fn empty(rank:usize)->Self{
		let buffer=Vec::new();
		let layout=Layout::new(vec![0;rank]);

		Self::from_inner(buffer,layout)
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
	/// reference the layout
	pub fn layout(&self)->&Layout{&self.layout}
	/// reference the layout
	pub fn layout_mut(&mut self)->&mut Layout{&mut self.layout}
	#[track_caller]
	/// create a new tensor. Err if the dims have a product greater than the data len
	pub fn new(data:Vec<E>,dims:impl AsRef<[usize]>)->Self{error::unwrap_or_panic(Self::try_new(data,dims))}
	/// swap out the buffer.  Note that producing a Tens with a layout invalid for its buffer is allowed, but may lead to incorrect behavior or panics on functions that assume layout validity
	pub fn replace_buffer(&mut self,mut replacement:Vec<E>)->Vec<E>{
		unsafe{		// safety: postcondition of Self::_from_raw_parts: When (ptr, len, cap) are not ok to put in Vec::from_raw_parts (borrowed buffer case), the resulting Tens must never convert its buffer to a slice or vec. Ensure all construction goes through _from_raw_parts.
			let old=Vec::from_raw_parts(mem::replace(&mut self.ptr,replacement.as_mut_ptr()),mem::replace(&mut self.len,replacement.len()),mem::replace(&mut self.cap,replacement.len()));
			mem::forget(replacement);

			old
		}
	}
	/// create a 0d tensor from a scalar
	pub fn scalar(data:E)->Self{
		let layout=Layout::new(Vec::new());
		Self::from_inner(vec![data],layout)
	}
	/// set the buffer. Note that producing a Tens with a layout invalid for its buffer is allowed, but may lead to incorrect behavior or panics on functions that assume layout validity
	pub fn set_buffer(&mut self,buffer:Vec<E>){
		self.replace_buffer(buffer);
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
	/// create a 1d tensor from a vector
	pub fn vector(data:Vec<E>)->Self{
		let layout=Layout::new([data.len()]);
		Self::from_inner(data,layout)
	}
}

/// A growable multidimensional array type, abbreviated from Tensor (Similar to Vec). Layout validity is lazily checked, and producing a Tens with a layout invalid for its buffer is allowed, but may lead to incorrect behavior or panics on functions that assume layout validity. The default value is an empty tensor with dims [0]
pub struct Tens<E>{
	layout:Layout,	// layout info, possibly modified
	ptr:*mut E,		// buffer pointer.
	len:usize,		// buffer length
	cap:usize,		// buffer capacity. 0 if not owned (when this is used as a View field). If nonzero, (ptr, len, cap) must be ok to convert to Vec
}
#[repr(transparent)]
/// A tensor value that operates in a functional style, with mostly chain-move functions rather than chain-mut functions. Layout validity is checked on construction. The default value is an empty tensor with dims [0]
pub struct Tensor<E>(Tens<E>);

use std::{
	borrow::{Borrow,BorrowMut},cmp::{Eq,PartialEq},hash::{Hash,Hasher},iter::FromIterator,mem,ops::{Deref,DerefMut,Index,IndexMut,Range},slice
};
use super::{
	Error,Layout,Position,Result,View,error,position::SignedIndexPosition,view::{ViewRef,ViewMut}
};
