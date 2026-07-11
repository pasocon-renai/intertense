impl<E> AsRef <View<E>>    for Tensor<E>{
	fn as_ref    (&    self)->&    View<E>{self.as_view()}
}
impl<E> AsMut <View<E>>    for Tensor<E>{
	fn as_mut    (&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E> Borrow<View<E>>    for Tensor<E>{
	fn borrow    (&    self)->&    View<E>{self.as_view()}
}
impl<E> BorrowMut<View<E>> for Tensor<E>{
	fn borrow_mut(&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E:Clone> Clone        for Tensor<E>{
	fn clone(&self)->Self{self.view().to_owned()}
	/*fn clone_from(&mut self,other:&Self)->Self{

	}*/
}
impl<E> Deref              for Tensor<E>{
	fn deref(&self)->&View<E>{self.as_view()}
	type Target=View<E>;
}
impl<E> DerefMut           for Tensor<E>{
	fn deref_mut(&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E> Drop for View<E>{
	fn drop(&mut self){
		let (ptr,len,cap)=(self.ptr,self.len,self.cap);
		unsafe{			// for safety, a valid buffer is maintained when cap>0. If cap==0, this is a field of a borrowed view and should not be dropped.
			if cap==0{return}
			mem::drop(Vec::from_raw_parts(ptr,len,cap));
		}
	}
}
impl<E:Eq> Eq for View<E>{}
impl<E:Eq> Eq for Tensor<E>{}
impl<E,P:SignedIndexPosition,const N:usize> Index<[P;N]> for View<E>{
	#[track_caller]
	fn index(&self,index:[P;N])->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E,P:SignedIndexPosition,const N:usize> Index<[P;N]> for Tensor<E>{
	#[track_caller]
	fn index(&self,index:[P;N])->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E> Index<Position> for View<E>{
	#[track_caller]
	fn index(&self,index:Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E> Index<Position> for Tensor<E>{
	#[track_caller]
	fn index(&self,index:Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E> Index<&Position> for View<E>{
	#[track_caller]
	fn index(&self,index:&Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E> Index<&Position> for Tensor<E>{
	#[track_caller]
	fn index(&self,index:&Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E,P:SignedIndexPosition> Index<&[P]> for Tensor<E>{
	#[track_caller]
	fn index(&self,index:&[P])->&Self::Output{self.as_view().index(index)}
	type Output=E;
}
impl<E,P:SignedIndexPosition> Index<&[P]> for View<E>{
	#[track_caller]
	fn index(&self,index:&[P])->&Self::Output{
		let i=self.layout.compute_offset(index);
		self.data().index(i)
	}
	type Output=E;
}
impl<E,P:SignedIndexPosition,const N:usize> IndexMut<[P;N]> for View<E>{
	#[track_caller]
	fn index_mut(&mut self,index:[P;N])->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E,P:SignedIndexPosition,const N:usize> IndexMut<[P;N]> for Tensor<E>{
	#[track_caller]
	fn index_mut(&mut self,index:[P;N])->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E> IndexMut<Position> for View<E>{
	#[track_caller]
	fn index_mut(&mut self,index:Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E> IndexMut<Position> for Tensor<E>{
	#[track_caller]
	fn index_mut(&mut self,index:Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E> IndexMut<&Position> for View<E>{
	#[track_caller]
	fn index_mut(&mut self,index:&Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E> IndexMut<&Position> for Tensor<E>{
	#[track_caller]
	fn index_mut(&mut self,index:&Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E,P:SignedIndexPosition> IndexMut<&[P]> for Tensor<E>{
	#[track_caller]
	fn index_mut(&mut self,index:&[P])->&mut Self::Output{self.as_mut_view().index_mut(index)}
}
impl<E,P:SignedIndexPosition> IndexMut<&[P]> for View<E>{
	#[track_caller]
	fn index_mut(&mut self,index:&[P])->&mut Self::Output{
		let i=self.layout.compute_offset(index);
		self.data_mut().index_mut(i)
	}
}
impl<E:PartialEq<X>,X> PartialEq<Tensor<X>> for Tensor<E>{
	fn eq(&self,other:&Tensor<X>)->bool{self.dims()==other.dims()&&self.positions().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&Tensor<X>)->bool{self.dims()!=other.dims()||self.positions().any(|ix|self[&ix]!=other[&ix])}
}
impl<E:PartialEq<X>,X> PartialEq<View<X>> for Tensor<E>{
	fn eq(&self,other:&View  <X>)->bool{self.dims()==other.dims()&&self.positions().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&View  <X>)->bool{self.dims()!=other.dims()||self.positions().any(|ix|self[&ix]!=other[&ix])}
}
impl<E:PartialEq<X>,X> PartialEq<Tensor<X>> for View<E>{
	fn eq(&self,other:&Tensor<X>)->bool{self.dims()==other.dims()&&self.positions().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&Tensor<X>)->bool{self.dims()!=other.dims()||self.positions().any(|ix|self[&ix]!=other[&ix])}
}
impl<E:PartialEq<X>,X> PartialEq<View<X>> for View<E>{
	fn eq(&self,other:&View  <X>)->bool{self.dims()==other.dims()&&self.positions().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&View  <X>)->bool{self.dims()!=other.dims()||self.positions().any(|ix|self[&ix]!=other[&ix])}
}
impl<E:Clone> ToOwned for View<E>{
	fn to_owned(&self)->Tensor<E>{
		let buffer=self.data().to_vec();
		let layout=self.layout. clone();

		Tensor::from_inner(buffer,layout)
	}
	type Owned=Tensor<E>;
}

impl<E> Tensor<E>{
	/// reference as a View. the resulting View does not have a validity guarantee; it is as valid as the Tensor it was referenced from
	pub fn as_mut_view(&mut self)->&mut View<E>{&mut self.0}
	/// reference as a View. the resulting View does not have a validity guarantee; it is as valid as the Tensor it was referenced from
	pub fn as_view(&self)->&View<E>{&self.0}
	/// reference the dims. no particular validity guarantee is present for unsafe purposes, but most nontrival functionality still expects a Tensor's layout to be mutably valid for its buffer
	pub fn dims_mut(&mut self)->&mut [usize]{
		unsafe{			// this can be made safe because Tensor does not have a validity guarantee
			self.0.dims_mut()
		}
	}
	/// create a new tensor from the inner data without checking. Most nontrival functionality still expects a Tensor's layout to be mutably valid for its buffer. violating this may result in panics or unexpected behavior
	pub fn from_inner(buffer:Vec<E>,layout:Layout)->Self{
		unsafe{		// Tensor wrapper does not have a validity guarantee, so it's safe to not check
			Self(View::from_inner_unchecked(buffer,layout))
		}
	}
	#[track_caller]
	/// convert into a view, panicing if invalid
	pub fn into_view(self)->View<E>{
		match self.try_into_view(){
			Err(e)=>panic!("{e}"),
			Ok(x)=>x
		}
	}
	#[track_caller]
	/// create a new tensor. panics if the layout is invalid
	pub fn new(buffer:Vec<E>,dims:impl AsRef<[usize]>)->Self{Self(View::new(buffer,dims))}
	/// reference the strides. no particular validity guarantee is present for unsafe purposes, but most nontrival functionality still expects a Tensor's layout to be mutably valid for its buffer
	pub fn strides_mut(&mut self)->&mut [isize]{
		unsafe{			// this can be made safe because Tensor does not have a validity guarantee
			self.0.strides_mut()
		}
	}
	/// convert into a View, erroring if invalid
	pub fn try_into_view(self)->Result<View<E>>{
		self.0.layout.validate_mut(self.0.data().len()).map_err(|e|e.with_op("view"))?;
		Ok(self.0)
	}
	/// check validity of the view for a shared context
	pub fn validate(&self)->Result<()>{self.layout.validate(self.data().len())}
	/// reference as a View. the resulting View does not have a validity guarantee; it is as valid as the Tensor it was referenced from
	pub fn view(&self)->&View<E>{&self.0}
}
impl<E> View<E>{
	/// get as a ptr
	pub fn as_ptr(&self)->*const E{self.ptr}
	/// get as a ptr
	pub fn as_mut_ptr(&mut self)->*mut E{self.ptr}
	/// reference as a View. inherit validity guarantees from wrapping
	pub fn as_mut_view(&mut self)->&mut View<E>{self}
	/// reference as a View. inherit validity guarantees from wrapping
	pub fn as_view(&self)->&View<E>{self}
	/// count the number of components
	pub fn count(&self)->usize{self.layout.count()}
	/// reference the data
	pub fn data(&self)->&[E]{
		let (ptr,len)=(self.ptr,self.len);
		unsafe{			// for safety we maintain a valid slice of data
			slice::from_raw_parts(ptr,len)
		}
	}
	/// reference the data
	pub fn data_mut(&mut self)->&mut [E]{
		let (ptr,len)=(self.ptr,self.len);
		unsafe{			// for safety we maintain a valid slice of data
			slice::from_raw_parts_mut(ptr,len)
		}
	}
	/// reference the dims
	pub fn dims(&self)->&[usize]{self.layout.dims()}
	/// reference the dims. the caller must maintain any applicable validity guarantees of the View Layout
	pub unsafe fn dims_mut(&mut self)->&mut [usize]{self.layout.dims_mut()}
	/// create a new View from a buffer and layout. the caller must uphold any applicable validity guarantees.
	pub unsafe fn from_inner_unchecked(mut buffer:Vec<E>,layout:Layout)->Self{
		let ptr=buffer.as_mut_ptr();
		let len=buffer.len();
		let cap=buffer.capacity();

		Self{layout,ptr,len,cap}
	}
	/// get the layout
	pub fn get_layout(&self)->Layout{self.layout.clone()}
	/// reference the layout
	pub fn layout(&self)->&Layout{&self.layout}
	/// reference the layout. the caller must maintain any applicable validity guarantees of the View Layout
	pub unsafe fn layout_mut(&mut self)->&mut Layout{&mut self.layout}
	/// wrap in a Tensor
	pub fn into_tensor(self)->Tensor<E>{Tensor(self)}
	/// get the len of the layout (may be less than the len of the data)
	pub fn len(&self)->usize{self.layout.len()}
	#[track_caller]
	/// create a new View from a buffer and dims, panicing for a mutable view
	pub fn new(buffer:Vec<E>,dims:impl AsRef<[usize]>)->Self{
		match Self::try_new(buffer,dims){
			Err(e)=>panic!("{e}"),
			Ok(x)=>x
		}
	}
	/// iterate over positions in a tensor
	pub fn positions(&self)->PositionIter{PositionIter::new(self.dims())}
	/// set the layout. will panic if invalid for a mutable view of the buffer
	pub fn set_layout(&mut self,layout:Layout){
		match layout.validate_mut(self.data().len()){
			Err(e)=>panic!("{e}"),
			Ok(_)=>self.layout=layout
		}
	}
	/// reference the strides
	pub fn strides(&self)->&[isize]{self.layout.strides()}
	/// reference the strides. the caller must maintain any applicable validity guarantees of the View Layout
	pub unsafe fn strides_mut(&mut self)->&mut [isize]{self.layout.strides_mut()}
	/// create a new View from a buffer and layout, erroring if invalid for a mutable view
	pub fn try_from_inner(buffer:Vec<E>,layout:Layout)->Result<Self>{
		layout.validate_mut(buffer.len())?;
		unsafe{		// the layout is safe because it has the strongest validity condition
			Ok(Self::from_inner_unchecked(buffer,layout))
		}
	}
	/// create a new View from a buffer and dims, erroring if invalid for a mutable view
	pub fn try_new(buffer:Vec<E>,dims:impl AsRef<[usize]>)->Result<Self>{
		let buffer=buffer.into();
		let layout=Layout::try_new(dims)?;

		Self::try_from_inner(buffer,layout).map_err(|e|e.with_op("new"))
	}
	#[track_caller]
	/// swap dims
	pub fn swap_dims(mut self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->Self{
		unsafe{			// swap_dims shouldn't affect layout validity
			self.layout_mut().swap_dims(a,b);
			self
		}
	}
	/// check validity of the view for a shared context
	pub fn validate_shared(&self)->Result<()>{self.layout.validate(self.data().len())}
	/// check validity of the view for a mutable context
	pub fn validate_mut(&self)->Result<()>{self.layout.validate_mut(self.data().len())}
	/// set the layout. will panic if invalid for a mutable view of the buffer
	pub fn with_layout(mut self,layout:Layout)->Self{
		self.set_layout(layout);
		self
	}
}

#[repr(transparent)]
/// owned tensor view. layout validity is not guaranteed, but maining appropriate layout validity is recommended. Failing to do so may result in panics or unexpected behavior. Tensor generally has in-place operations. use View for chain-moved operations and ViewRef/ViewMut for by-ref operations
pub struct Tensor<E>(View<E>);
/// tensor view that may have a validity guarantee from its wrapper. Where Tensor generally has in-place operations, View generally has chain-moved operations
/// validity guarantees:
/// Tensor wrapper does not guarantee validity
/// ViewRef wrapper guarantees shared validity
/// ViewMut wrapper guarantees mutable validity
/// An owned View created by safe code guarantees mutable validity
/// Due to the presence validity guarantees in some cases but not others, View<E> cannot implement Clone. use ToOwned instead or use Tensor for a clonable owned tensor, or ViewRef for a cheaply clonable reference type
pub struct View<E>{	// do not make View Clone because that would allow safe creation of not mutably valid View from ViewRef deref. use ToOwned instead
	layout:Layout,	// layout info, possibly modified
	ptr:*mut E,		// buffer pointer. ptr and len must always form a valid slice that makes the layout valid where layout validity is guaranteed
	len:usize,		// buffer length
	cap:usize,		// buffer capacity. 0 if not owned (when this is used as a ViewRef field)
}

use std::{
	borrow::{Borrow,BorrowMut,ToOwned},cmp::{Eq,PartialEq},error::Error as StdError,hash::{Hash,Hasher},iter::FromIterator,marker::PhantomData,mem,ops::{Deref,DerefMut,Index,IndexMut,Range},ptr,slice,result::Result as StdResult
};
use super::{Error,Layout,PositionIter,Position,Result,position::SignedIndexPosition};
