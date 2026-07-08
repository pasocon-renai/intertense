impl<   E> AsMut<Self>        for View<E>{
	fn as_mut    (&mut self)->&mut Self{self}
}
impl<'a,E> AsMut<Self>        for ViewMut<'a,E>{
	fn as_mut    (&mut self)->&mut Self{self}
}
impl<'a,E> AsMut<Self>        for ViewRef<'a,E>{
	fn as_mut    (&mut self)->&mut Self{self}
}
impl<'a,E> AsMut<View<E>>     for ViewMut<'a,E>{
	fn as_mut    (&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E>    AsRef<Self>        for View<E>{
	fn as_ref    (&    self)->&Self{self}
}
impl<'a,E> AsRef<Self>        for ViewMut<'a,E>{
	fn as_ref    (&    self)->&Self{self}
}
impl<'a,E> AsRef<Self>        for ViewRef<'a,E>{
	fn as_ref    (&    self)->&Self{self}
}
impl<'a,E> AsRef    <View<E>> for ViewMut<'a,E>{
	fn as_ref    (&    self)->&    View<E>{self.as_view()}
}
impl<'a,E> AsRef    <View<E>> for ViewRef<'a,E>{
	fn as_ref    (&    self)->&    View<E>{self.as_view()}
}
impl<'a,E> Borrow   <View<E>> for ViewMut<'a,E>{
	fn borrow    (&    self)->&    View<E>{self.as_view()}
}
impl<'a,E> Borrow   <View<E>> for ViewRef<'a,E>{
	fn borrow    (&    self)->&    View<E>{self.as_view()}
}
impl<'a,E> BorrowMut<View<E>> for ViewMut<'a,E>{
	fn borrow_mut(&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<'a,E:Clone> Clone        for ViewMut<'a,E>{
	fn clone(&self)->Self{							  	// new owned buffer with vc=1, so the mut view is valid
		Self{inner:self.to_tensor(),phantom:PhantomData}
	}
}
impl<'a,E      > Clone        for ViewRef<'a,E>{
	fn clone(&self)->Self{
		unsafe{
			Self{inner:self.inner.clone_ref(),phantom:PhantomData}
		}
	}
}
impl<'a,E      > Deref        for ViewMut<'a,E>{
	fn deref    (&    self)->&    Self::Target{self.as_view()}
	type Target=View<E>;
}
impl<'a,E      > Deref        for ViewRef<'a,E>{
	fn deref    (&    self)->&    Self::Target{self.as_view()}
	type Target=View<E>;
}
impl<'a,E      > DerefMut     for ViewMut<'a,E>{
	fn deref_mut(&mut self)->&mut Self::Target{self.as_mut_view()}
}
impl<   E:Eq> Eq   for View<E>{}
impl<'a,E:Eq> Eq   for ViewMut<'a,E>{}
impl<'a,E:Eq> Eq   for ViewRef<'a,E>{}
impl<E:Clone        > From<       &[E  ] > for View<E>{
	fn from(inner:&[E  ])->Self{Tensor::from_flat(inner).into()}
}
impl<E,const N:usize> From<[E;N]> for View<E>{
	fn from(inner: [E;N])->Self{Tensor::from_vec(inner).into()}
}
impl<      E            > From< Tensor <   E>> for View<E>{
	fn from(inner: Tensor <   E>)->Self{inner        .into_view()}
}
impl<      E:      Clone> From<&Tensor <   E>> for View<E>{
	fn from(inner:&Tensor <   E>)->Self{inner.clone().into_view()}
}
impl<      E            > From< Vec    <   E>> for View<E>{
	fn from(inner: Vec    <   E>)->Self{Tensor::from_vec(inner).into()}
}
impl<      E:      Clone> From<&View   <   E>> for View<E>{
	fn from(inner:&View   <   E>)->Self{inner.to_owned().into_view()}
}
impl<'a,   E:'a+   Clone> From< ViewMut<'a,E>> for View<E>{
	fn from(inner: ViewMut<'a,E>)->Self{inner        .into_view()}
}
impl<'a,   E:'a+   Clone> From<&ViewMut<'a,E>> for View<E>{
	fn from(inner:&ViewMut<'a,E>)->Self{inner.clone().into_view()}
}
impl<'a,'b,E:'a+'b+Clone> From<&ViewMut<'a,E>> for ViewMut<'b,E>{
	fn from(inner:&ViewMut<'a,E>)->Self{inner.clone().into_view_mut()}
}
impl<'a,'b,E:'a+'b+Clone> From<&ViewMut<'a,E>> for ViewRef<'b,E>{
	fn from(inner:&ViewMut<'a,E>)->Self{inner.clone().into_view_ref()}
}
impl<'a,   E:'a+   Clone> From< ViewRef<'a,E>> for View<E>{
	fn from(inner: ViewRef<'a,E>)->Self{inner        .into_view()}
}
impl<'a,   E:'a+   Clone> From<&ViewRef<'a,E>> for View<E>{
	fn from(inner:&ViewRef<'a,E>)->Self{inner.clone().into_view()}
}
impl<'a,'b,E:'a+'b+Clone> From<&ViewRef<'a,E>> for ViewMut<'b,E>{
	fn from(inner:&ViewRef<'a,E>)->Self{inner.clone().into_view_mut()}
}
impl<E> FromIterator<E> for View<E>{
	fn from_iter<I:IntoIterator<Item=E>>(iter:I)->Self{Tensor::from_vec(iter).into()}
}
impl<   E:Hash> Hash for View<E>{
	fn hash<H:Hasher>(&self,hasher:&mut H){self.0.hash(hasher)}
}
impl<'a,E:Hash> Hash for ViewMut<'a,E>{
	fn hash<H:Hasher>(&self,hasher:&mut H){(**self).hash(hasher)}
}
impl<'a,E:Hash> Hash for ViewRef<'a,E>{
	fn hash<H:Hasher>(&self,hasher:&mut H){(**self).hash(hasher)}
}
impl<E> Index   <&Position>      for View<E>{
	fn index(&self,index:&Position)->&Self::Output{self.0.index(index)}
	type Output=E;
}
impl<E> Index   < Position>      for View<E>{
	fn index(&self,index: Position)->&Self::Output{self.0.index(index)}
	type Output=E;
}
impl<  E,I:Copy+TryInto<isize>              > Index<&[I]  > for View<E>{
	fn index(&self,index:&[I])->&Self::Output{self.0.index(index)}
	type Output=E;
}
impl<   E,I:     TryInto<isize>,const N:usize> Index< [I;N]> for View<E>{
	fn index(&self,index:[I;N])->&Self::Output{self.0.index(index)}
	type Output=E;
}
impl<'a,E,I> Index<I> for ViewMut<'a,E> where View<E>:Index<I,Output=E>{
	fn index(&self,index:I)->&Self::Output{self.inner.view().index(index)}
	type Output=E;
}
impl<'a,E,I> Index<I> for ViewRef<'a,E> where View<E>:Index<I,Output=E>{
	fn index(&self,index:I)->&Self::Output{self.inner.view().index(index)}
	type Output=E;
}
impl<   E  > IndexMut<&Position>      for View<E>{
	fn index_mut(&mut self,index:&Position)->&mut Self::Output{self.0.index_mut(index)}
}
impl<   E  > IndexMut< Position>      for View<E>{
	fn index_mut(&mut self,index: Position)->&mut Self::Output{self.0.index_mut(index)}
}
impl<   E,I:Copy+TryInto<isize>>               IndexMut<&[I]>   for View<E>{
	fn index_mut(&mut self,index:&[I]     )->&mut Self::Output{self.0.index_mut(index)}
}
impl<   E,I:     TryInto<isize>,const N:usize> IndexMut< [I;N]> for View<E>{
	fn index_mut(&mut self,index: [I;N]   )->&mut Self::Output{self.0.index_mut(index)}
}
impl<'a,E,I> IndexMut< I> for ViewMut<'a,E> where View<E>:IndexMut<I,Output=E>{
	fn index_mut(&mut self,index: I       )->&mut Self::Output{self.as_mut_view().index_mut(index)}
}
impl<   E:PartialEq<X>,X> PartialEq<Tensor <   X>> for View<E>{
	fn eq(&self,other:&Tensor<X>)->bool{self.dims()==other.dims()&&self.indices().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&Tensor<X>)->bool{self.dims()!=other.dims()||self.indices().any(|ix|self[&ix]!=other[&ix])}
}
impl<   E:PartialEq<X>,X> PartialEq<View   <   X>> for View<E>{
	fn eq(&self,other:&View<X>)->bool{self.0.dims()==other.dims()&&self.indices().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&View<X>)->bool{self.0.dims()!=other.dims()||self.indices().any(|ix|self[&ix]!=other[&ix])}
}
impl<'a,E:PartialEq<X>,X> PartialEq<ViewMut<'a,X>> for ViewMut<'a,E>{
	fn eq(&self,other:&ViewMut<'a,X>)->bool{(**self).eq(&**other)}
	fn ne(&self,other:&ViewMut<'a,X>)->bool{(**self).ne(&**other)}
}
impl<'a,E:PartialEq<X>,X> PartialEq<ViewRef<'a,X>> for ViewMut<'a,E>{
	fn eq(&self,other:&ViewRef<'a,X>)->bool{(**self).eq(&**other)}
	fn ne(&self,other:&ViewRef<'a,X>)->bool{(**self).ne(&**other)}
}
impl<'a,E:PartialEq<X>,X> PartialEq<ViewMut<'a,X>> for ViewRef<'a,E>{
	fn eq(&self,other:&ViewMut<'a,X>)->bool{(**self).eq(&**other)}
	fn ne(&self,other:&ViewMut<'a,X>)->bool{(**self).ne(&**other)}
}
impl<'a,E:PartialEq<X>,X> PartialEq<ViewRef<'a,X>> for ViewRef<'a,E>{
	fn eq(&self,other:&ViewRef<'a,X>)->bool{(**self).eq(&**other)}
	fn ne(&self,other:&ViewRef<'a,X>)->bool{(**self).ne(&**other)}
}
impl<   E:Clone> ToOwned  for View<E>{
	fn to_owned(&self)->Self::Owned{self.to_tensor()}
	type Owned=Tensor<E>;
}

impl<E> View<E>{
	/// append the view to this one
	pub fn append(mut self,mut b:View<E>,i:impl TryInto<isize>)->Self{
		self.0.append(&mut b.0,i);
		self
	}
	/// get a pointer to the data buffer
	pub fn as_mut_ptr   (&mut self)->*mut            E {self.0.as_mut_ptr()}
	/// convert to a mutable view reference
	pub fn as_mut_view  (&mut self)->&mut View   <   E>{self.0.as_mut_view()}
	/// get a pointer to the data buffer
	pub fn as_ptr       (&    self)->*const          E {self.0.as_ptr()}
	/// convert to a view reference
	pub fn as_view      (&    self)->&    View   <   E>{self.0.as_view()}
	#[track_caller]
	/// convert to a viewmut reference
	pub fn as_view_mut  (&mut self)->&mut ViewMut<'_,E>{self.0.as_view_mut()}
	#[track_caller]
	/// convert to a viewref reference
	pub fn as_view_ref  (&    self)->&    ViewRef<'_,E>{self.0.as_view_ref()}
	#[track_caller]
	/// apply broadcast dim to each index. errors if dims.len()!=self.rank(), or if any of the individual dims fail to broadcast
	pub fn broadcast    (     self,dims:impl AsRef<[usize]>)          ->Self where E:Clone{self.into_view_mut().broadcast(dims).into_view()}
	#[track_caller]
	/// broadcast dim. If the index is out of bounds of the rank, the result is an invalid index error. If dim at the index is 1 and requested size is not 1, the result is a view with the dim equal to the requested size, accomplished by index aliasing the components. If the dim at index is neither 1 nor size, and size is not 1, the result is a mismatch error. If the dim at index is size or size is 1, the result is unchanged from the input
	pub fn broadcast_dim(     self,axis:impl TryInto<isize>,dim:usize)->Self where E:Clone{self.into_view_mut().broadcast_dim(axis.try_into().unwrap_or(isize::MIN),dim).into_view()}
	/// reference the underlying buffer
	pub fn buffer       (&    self)->&[E]    {self.0.buffer()}
	/// get the capacity of the underlying buffer
	pub fn buffer_cap   (&    self)->usize   {self.0.buffer_cap()}
	/// get the length of the underlying buffer
	pub fn buffer_len   (&    self)->usize   {self.0.buffer_len()}
	/// reference the underlying buffer
	pub fn buffer_mut   (&mut self)->&mut [E]{self.0.buffer_mut()}
	/// check if append would succeed
	pub fn check_append<I:TryInto<isize>>(&self,b:&Tensor<E>,index:I)->Result<()> where I::Error:'static+StdError{self.0.check_append(b,index)}
	/// check if broadcast would succeed
	pub fn check_broadcast(&self,dims:&[usize])->Result<()>{self.view_ref().try_broadcast(dims).map(|_|())}
	/// check if broadcast would succeed
	pub fn check_broadcast_dim<I:TryInto<isize>>(&self,index:I,rhsdim:usize)->Result<()> where I::Error:'static+StdError{self.view_ref().try_broadcast_dim(index,rhsdim).map(|_|())}
	/// check if flip dim would succeed
	pub fn check_flip_dim     <I:TryInto<isize>>(&self,index:I)             ->Result<()> where I::Error:'static+StdError{self.view_ref().try_flip_dim(index).map(|_|())}
	/// normalizes the layout and removes excess capacity
	pub fn compactify(&mut self){self.0.compactify()}
	/// counts the number of components
	pub fn count(&self)->  usize {self.0.count()}
    /// reference the dimensions
	pub fn dims (&self)->&[usize]{self.0.dims()}
	/// reverse the order of components along the specified axis
	pub fn flip_dim(mut self,index:impl TryInto<isize>)->Self{
		*self.0.layout_mut()=self.view_ref().flip_dim(index).get_layout();
		self
	}
	/// reverse the order of components along all axes
	pub fn flip    (mut self)->Self{
		self.0.strides_mut().iter_mut().for_each(|s|*s=-*s);
		self
	}
	/// creates tensor from raw parts:
	/// capacity: capacity of the underlying buffer
	/// ptr: pointer to the buffer
	/// layout: valid pointer to a layout
	/// len: length of the underlying buffer
	pub unsafe fn from_raw_parts(layout:*const Layout,ptr:*mut E,len:usize,cap:usize,vc:*const ())->Self{
		unsafe{Self(Tensor::from_raw_parts(layout,ptr,len,cap,vc))}
	}
	/// convert from a tensor
	pub fn from_tensor(inner:impl Into<Tensor<E>>)->Self{Self(inner.into())}
	/// convert from a view
	pub fn from_view  (inner:impl Into<View  <E>>)->Self{inner.into()}
	/// get the layout
	pub fn get_layout (&self)->Layout   {self.0.get_layout()}
	/// returns an iterator over the view indices
	pub fn indices    (&self)->GridIter {self.0.indices()}
	/// convert into a tensor
	pub fn into_tensor( self)->Tensor<E>{self.0}
	/// convert into a view
	pub fn into_view  ( self)->View  <E>{self}
	/// convert into a view
	pub fn into_view_mut<'a>   ( self)->ViewMut<'a,E> where E:'a{self.0.into_view_mut()}
	/// convert into a view
	pub fn into_view_ref<'a>   ( self)->ViewRef<'a,E> where E:'a{self.0.into_view_ref()}
	/// check if more than owned 1 tensor or ref with the buffer
	pub fn is_borrowed         (&self)             ->bool{self.0.view_count()>1}
	/// checks if the view is empty
	pub fn is_empty            (&self)             ->bool{self.0.is_empty()}
	/// checks if the layout is normalized
	pub fn is_layout_normalized(&self)             ->bool{self.0.is_layout_normalized()}
	/// checks if the layout is valid for the given mutability flag
	pub fn is_layout_valid     (&self,mutable:bool)->bool{self.0.layout().is_valid_for(mutable,self.buffer_len())}
	/// check if only 1 owned tensor or ref with the buffer
	pub fn is_unique  (&self)->bool{self.0.view_count()==1}
	/// reference the layout
	pub fn layout     (&self)->&Layout  {self.0.layout()}
	#[track_caller]
	/// map the components by reference
	pub fn map              <F:FnMut(&     E               )->Y,  Y>(&    self,mut f:F)->Tensor<Y>{
		let data=self.indices().map(|ix|f(&self[ix])).collect();
		let dims=self.dims().to_vec();

		Tensor::new(data,dims)
	}
	#[track_caller]
	/// maps the components by reference
	pub fn map_2            <F:FnMut(&     E,&    X        )->Y,X,Y>(&    self,mut f:F,    x:impl AsRef<View<X>>)->Tensor<Y>{
		let x=x.as_ref().as_view_ref().try_broadcast(self.dims()).unwrap();

		let dims=x.dims().to_vec();
		let this=self.view_ref().try_broadcast(x.dims()).unwrap();

		let data=this.indices().map(|ix|f(&this[&ix],&x[&ix])).collect();

		Tensor::new(data,dims)
	}
	#[track_caller]
	/// maps the components by reference
	pub fn map_2_mut        <F:FnMut(&mut E,&mut X         )->Y,X,Y>(&mut self,mut f:F,mut x:impl AsMut<View<X>>)->Tensor<Y> where E:Clone,X:Clone{
		let mut x=x.as_mut().view_mut().try_broadcast(self.dims()).unwrap();

		let dims=x.dims().to_vec();
		let mut this=self.view_mut().try_broadcast(x.dims()).unwrap();

		let data=this.indices().map(|ix|f(&mut this[&ix],&mut x[&ix])).collect();

		Tensor::new(data,dims)
	}
	#[track_caller]
	/// maps the components by reference
	pub fn map_2_indexed    <F:FnMut(&    E,&    X,Position)->Y,X,Y>(&    self,mut f:F,    x:impl AsRef<View<X>>)->Tensor<Y>{
		let x=x.as_ref().as_view_ref().try_broadcast(self.dims()).unwrap();

		let dims=x.dims().to_vec();
		let this=self.view_ref().try_broadcast(x.dims()).unwrap();

		let data=this.indices().map(|ix|f(&this[&ix],&x[&ix],ix)).collect();

		Tensor::new(data,dims)
	}
	#[track_caller]
	/// maps the components by reference
	pub fn map_2_indexed_mut<F:FnMut(&mut E,&mut X,Position)->Y,X,Y>(&mut self,mut f:F,mut x:impl AsMut<View<X>>)->Tensor<Y> where E:Clone,X:Clone{
		let mut x=x.as_mut().view_mut().try_broadcast(self.dims()).unwrap();

		let dims=x.dims().to_vec();
		let mut this=self.view_mut().try_broadcast(x.dims()).unwrap();

		let data=this.indices().map(|ix|f(&mut this[&ix],&mut x[&ix],ix)).collect();

		Tensor::new(data,dims)
	}
	#[track_caller]
	/// maps the components by reference
	pub fn map_indexed      <F:FnMut(&    E,       Position)->Y,  Y>(&    self,mut f:F)->Tensor<Y>{
		let data=self.indices().map(|ix|f(&self[&ix],ix)).collect();
		let dims=self.dims().to_vec();

		Tensor::new(data,dims)
	}
	#[track_caller]
	/// maps the components by reference
	pub fn map_indexed_mut<F:FnMut(&mut E,         Position)->Y,Y>(&mut self,mut f:F)->Tensor<Y>{
		let data=self.indices().map(|ix|f(&mut self[&ix],ix)).collect();
		let dims=self.dims().to_vec();

		Tensor::new(data,dims)
	}
	#[track_caller]
	/// map the components by reference
	pub fn map_mut        <F:FnMut(&mut E                  )->Y,Y>(&mut self,mut f:F)->Tensor<Y>{
		let data=self.indices().map(|ix|f(&mut self[ix])).collect();
		let dims=self.dims().to_vec();

		Tensor::new(data,dims)
	}
	/// return the number of axes in the tensor
	pub fn rank(&self)->usize{self.0.rank()}
	#[track_caller]
	/// set the dims to a new dims with the same count. errors if the resulting view would have invalid layout. may dissociate from the original tensor
	pub fn reshape    (     self,dims: impl AsRef<[usize]>)->Self where E:Clone{self.into_view_mut().reshape(dims).into_view()}
	#[track_caller]
	/// set the layout. errors if the new layout would be invalid for a mutable view with this view's buffer
	pub fn set_layout (&mut self,layout:Layout){self.0.set_layout(layout)}
	#[track_caller]
	/// slice
	pub fn slice<I:Copy+TryInto<isize>>(self,ranges:impl AsRef<[Range<I>]>)->Self{
		impl<T:TryInto<isize>> TryInto<isize> for TryIntoIsizeMin<T>{
			fn try_into(self)->Result<isize>{Ok(self.0.try_into().unwrap_or(isize::MIN))}
			type Error=Error;
		}

		#[derive(Clone,Copy,Debug)]
		#[repr(transparent)]
		struct TryIntoIsizeMin<T:TryInto<isize>>(T);

		let ranges:&[Range<TryIntoIsizeMin<I>>]=unsafe{mem::transmute(ranges.as_ref())};
		self.try_slice(ranges).unwrap()
	}
	#[track_caller]
	/// slice
	pub fn slice_dim<I:TryInto<isize>,J:TryInto<isize>>(self,index:I,range:Range<J>)->Self{self.try_slice_dim(index.try_into().unwrap_or(isize::MIN),range.start.try_into().unwrap_or(isize::MIN)..range.end.try_into().unwrap_or(isize::MIN)).unwrap()}
	/// reference the strides
	pub fn strides    (&    self)->&[isize]    {self.0.layout().strides()}
	#[track_caller]
	/// swap two axes of the tensor. errors if invalid indices
	pub fn swap_dims  (     self,a:impl TryInto<isize>,b:impl TryInto<isize>)->Self{self.into_view_ref().swap_dims  (a,b).inner.into()}
	#[track_caller]
	/// squeeze a 1 dim of the tensor into nonexistence. errors if invalid indices or the dim is not 1
	pub fn squeeze_dim(     self,a:impl TryInto<isize>                      )->Self{self.into_view_ref().squeeze_dim(a, ).inner.into()}
	#[track_caller]
	/// flattens the view into a vec
	pub fn to_flat_vec  (&self,mem:impl Into<Option<Vec<E>>>)->Vec<E> where E:Clone{
		let mut mem=mem.into().unwrap_or_default();
		if self.0.is_layout_normalized(){
			let len=self.dims()[0]*self.strides()[0].abs() as usize;
			mem.extend_from_slice(unsafe{slice::from_raw_parts(self.as_ptr(),len)});
		}else{
			let mut position=&mut vec![0;self.rank()];

			mem.reserve(self.count());
			index::for_positions(self.dims(),|ix|mem.push(self[&*ix].clone()),&mut position);
		}

		mem
	}
	#[track_caller]
	/// convert the view to an owned tensor
	pub fn to_tensor  (&self)->Tensor<E> where E:Clone{
		let data=self.to_flat_vec(None);
		let dims=self.dims().to_vec();

		Tensor::new(data,dims)
	}
	#[track_caller]
	/// dissociate into a unique ref
	pub fn to_view_mut<'a>(&self)->ViewMut<'a,E> where E:'a+Clone{
		let mut inner=self.to_owned();
		assert!(inner.view_mut().is_layout_valid(),"wrapped view references must have valid layouts");

		ViewMut{inner,phantom:PhantomData}
	}
	#[track_caller]
	/// dissociate into a unique ref
	pub fn to_view_ref<'a>(&self)->ViewRef<'a,E> where E:'a+Clone{
		let inner=self.to_owned();
		assert!(inner.view_ref().is_layout_valid(),"wrapped view references must have valid layouts");

		ViewRef{inner,phantom:PhantomData}
	}
	#[track_caller]
	/// fallible indexing. tries to reference the component, returning an error if the indexing fails.
	pub fn try_component    <'a,I:Copy+TryInto<isize>>(&'a     self,indices:&[I])->Result<&'a E> where I::Error:'static+StdError{self.0.try_component    (indices)}
	#[track_caller]
	/// fallible indexing. tries to reference the component, returning an error if the indexing fails.
	pub fn try_component_mut<'a,I:Copy+TryInto<isize>>(&'a mut self,indices:&[I])->Result<&'a E> where I::Error:'static+StdError{self.0.try_component_mut(indices)}
	#[track_caller]
	/// attempt to set the layout. errors if the new layout would be invalid for a mutable view with this view's buffer
	pub fn try_set_layout   (&mut self,layout:Layout)->Result<()>{
		layout.check_validity_for(true,self.buffer_len()).map_err(|e|e.with_op("set layout"))?;
		Ok(*self.0.layout_mut()=layout)
	}
	#[track_caller]
	/// slice
	pub fn try_slice    <            J:Copy+TryInto<isize>>(self,ranges:impl AsRef<[Range<J>]>)->Result<Self> where                           J::Error:'static+StdError{
		let this=self.into_view_ref();
		let (layout,offset)=unsafe{			// for code reuse and avoiding having to backshift multiple times, unsafely extract buffer offset info from view ref slice result
			let view=this.try_slice(ranges)?;
			let layout=view.get_layout();	// get the layout and offset. the two pointers are within the same buffer
			let offset=view.inner.as_ptr().offset_from(this.as_ptr()) as usize;

			(layout,offset)
		};

		let (mut buffer,_)=this.inner.into_inner();
											// trim the start of the buffer according to the new offset. trim the end of it too since this is a reasonable place to do it
		buffer.truncate(layout.len()+offset);
		buffer.drain(..offset);
											// reconstruct the view with its new layout
		Ok(Self(Tensor::from_inner(buffer,layout)))
	}
	#[track_caller]
	/// slice
	pub fn try_slice_dim<I:TryInto<isize>,J:TryInto<isize>>(self,index:I,slicerange:Range<J>)  ->Result<Self> where I::Error:'static+StdError,J::Error:'static+StdError{
		let this=self.into_view_ref();
		let (layout,offset)=unsafe{			// for code reuse and avoiding having to backshift multiple times, unsafely extract buffer offset info from view ref slice result
			let view=this.try_slice_dim(index,slicerange)?;
			let layout=view.get_layout();	// get the layout and offset. the two pointers are within the same buffer
			let offset=view.inner.as_ptr().offset_from(this.as_ptr()) as usize;

			(layout,offset)
		};

		let (mut buffer,_)=this.inner.into_inner();
											// trim the start of the buffer according to the new offset. trim the end of it too since this is a reasonable place to do it
		buffer.truncate(layout.len()+offset);
		buffer.drain(..offset);
											// reconstruct the view with its new layout
		Ok(Self(Tensor::from_inner(buffer,layout)))
	}
	#[track_caller]
	/// conjure a dim one size 1 into existence
	pub fn unsqueeze_dim(self,a:impl TryInto<isize>)->Self{self.into_view_ref().unsqueeze_dim(a).inner.into()}
	/// gets the vc pointer
	pub fn vc(&self)->*const (){self.0.vc()}
	/// reference this tensor as a view
	pub fn view       (&    self)->&    View<E>{self}
	#[track_caller]
	/// produce a ViewMut to this tensor
	pub fn view_mut   (&mut self)->     ViewMut<'_,E>{
		assert!(self.is_layout_valid(true));	// ensure layout validity for a mutable view
		unsafe{									// the result takes the lifetime of self, so it won't mutably alias the buffer. ViewMut is a transparent repr of Tensor
			ViewMut{inner:self.0.clone_ref(),phantom:PhantomData}
		}
	}
	#[track_caller]
	/// produce a ViewRef to this tensor
	pub fn view_ref   (&    self)->     ViewRef<'_,E>{
		assert!(self.is_layout_valid(false));	// ensure layout validity for a shared view
		unsafe{									// the result takes the lifetime of self, so it won't mutably alias the buffer. ViewRef is a transparent repr of Tensor
			ViewRef{inner:self.0.clone_ref(),phantom:PhantomData}
		}
	}
}
impl<'a,E:'a> ViewMut<'a,E>{
	#[track_caller]
	/// clones the components of other into self, concatenating along the specified axis. may dissociate from the original tensor. no broadcasting is performed. the operation will fail if the dimensions don't match or if the resulting tensor is too big
	pub fn append(self,mut b:impl AsMut<View<E>>,i:impl TryInto<isize>)->Self where E:Clone{
		let mut this=self.into_unique();
		this.inner.append(&mut b.as_mut().0,i);

		this
	}
	/// reference as a view
	pub fn as_mut_view(&mut self)->&mut View<E>{
		unsafe{mem::transmute(self)}
	}
	/// reference as a view
	pub fn as_view    (&    self)->&    View<E>{
		unsafe{mem::transmute(self)}
	}
	#[track_caller]
	/// apply broadcast dim to each index. errors if dims.len()!=self.rank(), or if any of the individual dims fail to broadcast. may dissociate from the original tensor.
	pub fn broadcast    (self,dims:impl AsRef<[usize]>)          ->Self where E:Clone{self.try_broadcast(dims).unwrap()}
	#[track_caller]
	/// broadcast dim. If the index is out of bounds of the rank, the result is an invalid index error. may dissociate from the original tensor. If dim at the index is 1 and requested size is not 1, the result is a view with the dim equal to the requested size, accomplished by index aliasing the components. If the dim at index is neither 1 nor size, and size is not 1, the result is a mismatch error. If the dim at index is size or size is 1, the result is unchanged from the input
	pub fn broadcast_dim(self,axis:impl TryInto<isize>,dim:usize)->Self where E:Clone{self.try_broadcast_dim(axis.try_into().unwrap_or(isize::MIN),dim).unwrap()}
	/// reverse the order of components along the specified axis
	pub fn flip_dim (mut self,index:impl TryInto<isize>)->Self{
		*self.inner.layout_mut()=self.view_ref().flip_dim(index).get_layout();
		self
	}
	/// reverse the order of components along all axes
	pub fn flip     (mut self)->Self{
		self.inner.strides_mut().iter_mut().for_each(|s|*s=-*s);
		self
	}
	/// convert from a tensor
	pub fn from_tensor(inner:impl Into<Tensor<E>>)->Self{
		Self{inner:inner.into(),phantom:PhantomData}
	}
	/// convert from a view
	pub fn from_view  (inner:impl Into<View<E>>)  ->Self{
		Self{inner:inner.into().0,phantom:PhantomData}
	}
	/// convert into the buffer, cloning if the view's access to the buffer is not unique
	pub fn into_buffer  (self)->Vec<E> where E:Clone{
		if self.is_unique(){self.inner.into_buffer()}else{self.inner.buffer().to_vec()}
	}
	/// convert into a flattened vec
	pub fn into_flat_vec(self,mem:impl Into<Option<Vec<E>>>)->Vec<E> where E:Clone{
		if self.is_unique(){self.inner.into_flat_vec(mem)}else{self.to_flat_vec(mem)}
	}
	/// convert into a tensor, dissociating from the tensor to become an independent allocation if not unique
	pub fn into_tensor      (mut self)->Tensor <   E> where E:   Clone{
		if self.is_borrowed(){self.inner=self.to_tensor()}
		unsafe{less_annoying_transmute(self)}
	}
	/// dissociate from the original tensor and clone the inner buffer into a new allocation if necessary
	pub fn into_unique<'b>( self)->ViewMut<'b,E> where E:'b+Clone{self.into_view_mut()}
	/// convert into a view, dissociating from the tensor to become an independent allocation if not unique
	pub fn into_view        (mut self)->View   <   E> where E:   Clone{
		if self.is_borrowed(){self.inner=self.to_tensor()}
		unsafe{less_annoying_transmute(self)}
	}
	/// convert into a view mut, dissociating from the tensor to become an independent allocation if not unique
	pub fn into_view_mut<'b>(mut self)->ViewMut<'b,E> where E:'b+Clone{
		if self.is_borrowed(){self.inner=self.to_tensor()}
		unsafe{less_annoying_transmute(self)}
	}
	/// convert into a view ref, dissociating from the tensor to become an independent allocation if not unique
	pub fn into_view_ref<'b>(mut self)->ViewRef<'b,E> where E:'b+Clone{
		if self.is_borrowed(){self.inner=self.to_tensor()}
		unsafe{less_annoying_transmute(self)}
	}/// checks if the layout is valid
	pub fn is_layout_valid  (&   self)->bool{self.inner.layout().is_valid_for(true,self.inner.buffer_len())}
	#[track_caller]
	/// map the tensor components in the buffer in an unspecified order. may include components that are in the buffer but not visible to indexing, compactify first or use map indexed if this is undesirable. result will not be associated with the original tensor
	pub fn map        <'b,F:FnMut(&mut E         )->Y,        Y:'b>(&mut self,mut f:F                      )->ViewMut<'b,Y>             {self.try_map::        <_,Error,  Y>(|e    |Ok(f(e    ))  ).unwrap()}
	#[track_caller]
	/// map the tensor components in the buffer in an unspecified order. may include components that are in the buffer but not visible to indexing, compactify first or use map indexed if this is undesirable. result will not be associated with the original tensor
	pub fn map_2      <'b,F:FnMut(&mut E,&mut X  )->Y,X:Clone,Y:'b>(&mut self,mut f:F,x:impl AsMut<View<X>>)->ViewMut<'b,Y>where E:Clone{self.try_map_2::      <_,Error,X,Y>(|x0,x1|Ok(f(x0,x1)),x).unwrap()}
	#[track_caller]
	/// map the tensor components in the buffer in order of a lex iteration over the indices. the current position is also passed to the function. result will not be associated with the original tensor
	pub fn map_indexed<'b,F:FnMut(&mut E,Position)->Y,        Y:'b>(&mut self,mut f:F                      )->ViewMut<'b,Y>             {self.try_map_indexed::<_,Error,  Y>(|e ,ix|Ok(f(e,ix ))  ).unwrap()}
	#[track_caller]
	/// reshape the tensor to the specified dims. errors if the component count differs. may dissociate from the original tensor
	pub fn reshape(self,dims:impl AsRef<[usize]>)->Self where E:Clone{self.try_reshape(dims).unwrap()}
	#[track_caller]
	/// slice
	pub fn slice<I:Copy+TryInto<isize>>(&mut self,ranges:impl AsRef<[Range<I>]>)->ViewMut<'_,E>{
		impl<T:TryInto<isize>> TryInto<isize> for TryIntoIsizeMin<T>{
			fn try_into(self)->Result<isize>{Ok(self.0.try_into().unwrap_or(isize::MIN))}
			type Error=Error;
		}

		#[derive(Clone,Copy,Debug)]
		#[repr(transparent)]
		struct TryIntoIsizeMin<T:TryInto<isize>>(T);

		let ranges:&[Range<TryIntoIsizeMin<I>>]=unsafe{mem::transmute(ranges.as_ref())};
		self.try_slice(ranges).unwrap()
	}
	#[track_caller]
	/// slice
	pub fn slice_dim<I:TryInto<isize>,J:TryInto<isize>>(&mut self,index:I,range:Range<J>)->ViewMut<'_,E>{self.try_slice_dim(index.try_into().unwrap_or(isize::MIN),range.start.try_into().unwrap_or(isize::MIN)..range.end.try_into().unwrap_or(isize::MIN)).unwrap()}
	#[track_caller]
	/// swap the axes
	pub fn swap_dims    (self,i   :impl TryInto<isize>,j:impl TryInto<isize>)->Self{self.try_swap_dims(i.try_into().unwrap_or(isize::MIN),j.try_into().unwrap_or(isize::MIN)).unwrap()}
	#[track_caller]
	/// try broadcasting
	pub fn try_broadcast(self,dims:impl AsRef<[usize]>)->Result <Self> where E:Clone{
		let dims=dims.as_ref();

		let (expected,rank)=(self.0.rank(),dims.len());
		if expected!=rank{
			return Err(Error::mismatch(expected,self.get_layout(),"broadcast",rank,None,"rank"));
		}
		(0..rank).try_fold(self,|acc,n|acc.try_broadcast_dim(n,dims[n])).map_err(|e|e.with_rhs_layout(Layout::from_inner(dims.to_vec(),Vec::new())))
	}
	#[track_caller]
	/// try broadcasting the specified dimension. to the size. If the index is out of bounds of the rank, the result is an invalid index error. If dim at the index is 1 and requested size is not 1, the result is a view with the dim equal to the requested size, accomplished by index aliasing the components. If the dim at index is neither 1 nor size, and size is not 1, the result is a mismatch error. If the dim at index is size or size is 1, the result is unchanged from the input
	pub fn try_broadcast_dim<I:TryInto<isize>>(self,axis:I,dim:usize)->Result<Self> where E:Clone,I::Error:'static+StdError{
		let axis=axis.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"broadcast",None))?;
		let rank=self.rank();
								// try to convert the index to a usize that can index into dims
		let axis=index::normalize_index(rank,axis).ok_or(Error::invalid_index(axis,self.get_layout(),"broadcast"))?;
		let current=self.dims()[axis];
								// if the dims are already equal or the broadcast request is 1, no broadcast is needed
		if current==dim||dim==1{return Ok(self)}
		if current!=1{				// if unequal and not 1, no broadcast is possible
			return Err(Error::mismatch(current,self.get_layout(),"broadcast",dim,None,"dim"))
		}
								// create a view ref to return
		let mut result=self.view_ref();
		let layout=result.inner.layout_mut();
								// set the dim of the view ref to the broadcasted size. stride=0 so moving along the broadcasted axis doesn't changed the referenced component
		layout.dims_mut()   [axis]=dim;
		layout.strides_mut()[axis]=0;
								// done
		Ok(result.into_view_mut())
	}
	/// reverse the order of components along the specified axis
	pub fn try_flip_dim     <I:TryInto<isize>>(mut self,index:I         )->Result<Self> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"flip",None))?;
		let rank=self.rank();
								// try to convert the index to a usize that can index into dims
		let index=index::normalize_index(rank,index).ok_or(Error::invalid_index(index,self.get_layout(),"flip"))?;

		self.inner.strides_mut()[index]*=-1;
		Ok(self)
	}
	#[track_caller]
	/// map the tensor components in the buffer in an unspecified order. may include components that are in the buffer but not visible to indexing, compactify first or use map indexed if this is undesirable. result is not associated with the original tensor, however components of the original tensor may be mutated by the function
	pub fn try_map        <'b,F:FnMut(&mut E         )->StdResult<Y,R>,R:'static+StdError,        Y:'b>(&mut self,mut f:F                      )->Result<ViewMut<'b,Y>>{
		// apparently we can't linearize this in general without ub.. probably could if the layout is normalized though
		self.try_map_indexed(|e,_ix|f(e))
	}
	#[track_caller]
	/// map the tensor components in the buffer in an unspecified order. may include components that are in the buffer but not visible to indexing, compactify first or use map indexed if this is undesirable. result is not associated with the original tensor, however the components in the original tensor may be cloned during broadcasting and mutated by the function
	pub fn try_map_2      <'b,F:FnMut(&mut E,&mut X  )->StdResult<Y,R>,R:'static+StdError,X:Clone,Y:'b>(&mut self,mut f:F,mut x:impl AsMut<View<X>>)->Result<ViewMut<'b,Y>> where E:Clone{
		let (ll,rl)=(self.get_layout(),x.as_mut().get_layout());
		let x0=self;
		let x1=x.as_mut();

		let (mut x0,mut x1)=(x0.view_mut().try_broadcast(rl.dims())?,x1.view_mut().try_broadcast(ll.dims())?);
		x0.try_map_indexed(|x,ix|f(x,&mut x1[ix])).map_err(|e|e.with_rhs_layout(rl))
	}
	#[track_caller]
	/// map the tensor components in the buffer in order of a lex iteration over the indices. the current position is also passed to the function. result is not associated with the original tensor, however, components of the original tensor may be mutated by the function
	pub fn try_map_indexed<'b,F:FnMut(&mut E,Position)->StdResult<Y,R>,R:'static+StdError,        Y:'b>(&mut self,mut f:F)->Result<ViewMut<'b,Y>>{
		let mut buffer=Vec::with_capacity(self.count());
		for ix in self.indices(){buffer.push(f(&mut self[&ix],ix).map_err(|e|Error::other(e,self.get_layout(),"map",None))?)}

		let layout=Layout::new(self.dims());
		Ok(ViewMut{inner:Tensor::from_inner(buffer,layout),phantom:PhantomData})
	}
	#[track_caller]
	/// attempt to reshape the tensor to the specified dims. errors if the component count differs. may dissociate from the original tensor
	pub fn try_reshape(self,dims:impl AsRef<[usize]>)->Result<Self> where E:Clone{
		let dims=dims.as_ref();
		let newcount=dims.iter().try_fold(1,|acc:usize,&item|acc.checked_mul(item)).ok_or_else(||Error::too_big(Some(self.buffer_len()),None,self.get_layout(),Some(self.layout().len()),"reshape"))?;

		if self.count()!=newcount{return Err(Error::mismatch(self.count(),self.get_layout(),"reshape",newcount,Layout::new(dims),"count"))}

		let mut this=self.clone();
		if !this.is_layout_normalized(){
			this=this.into_unique();
			this.inner.normalize_layout();
		}

		*this.inner.layout_mut()=Layout::new(dims);
		Ok(this)
	}
	#[track_caller]
	/// attempt to set the layout. errors if the new layout would be invalid for a mutable view with this view's buffer
	pub fn try_set_layout   (&mut self,layout:Layout)->Result<()>{
		layout.check_validity_for(true,self.buffer_len()).map_err(|e|e.with_op("set layout"))?;
		Ok(*self.inner.layout_mut()=layout)
	}
	#[track_caller]
	/// slice
	pub fn try_slice    <            J:Copy+TryInto<isize>>(&mut self,ranges:impl AsRef<[Range<J>]>)->Result<ViewMut<'_,E>> where                           J::Error:'static+StdError{
		unsafe{// TODO extract most of this into a helper function
			let info=self.view_ref().try_slice(ranges)?.inner;
			let layout=info.get_layout();
			let offset=info.as_ptr().offset_from(self.as_ptr()) as usize;

			mem::drop(info);

			let mut result=self.view_mut();

			*result.inner.layout_mut()=layout;
			result.inner.offset(offset);

			Ok(result)
		}
	}
	#[track_caller]
	/// slice
	pub fn try_slice_dim<I:TryInto<isize>,J:TryInto<isize>>(&mut self,index:I,slicerange:Range<J>  )->Result<ViewMut<'_,E>> where I::Error:'static+StdError,J::Error:'static+StdError{
		unsafe{
			let info=self.view_ref().try_slice_dim(index,slicerange)?.inner;
			let layout=info.get_layout();
			let offset=info.as_ptr().offset_from(self.as_ptr()) as usize;

			mem::drop(info);

			let mut result=self.view_mut();

			*result.inner.layout_mut()=layout;
			result.inner.offset(offset);

			Ok(result)
		}
	}
	#[track_caller]
	/// attempt to return an axis of dim 1 to nothingness, erroring if the dim is not 1 or the index is out of bounds
	pub fn try_squeeze_dim<I:TryInto<isize>>(self,index       :I    )->Result<Self> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"squeeze",None))?;
		let index=index::normalize_index(self.rank(),index).ok_or(Error::invalid_index(index,self.get_layout(),"squeeze"))?;

		let dim=self.dims()[index];
		if dim!=1{return Err(Error::mismatch(dim,self.get_layout(),"squeeze",1,None,"dim"))}

		let mut this=self;
		(this.inner.dims_mut().remove(index),this.inner.strides_mut().remove(index));

		Ok(this)
	}
	#[track_caller]
	/// swap the axes if in bounds
	pub fn try_swap_dims  <I:TryInto<isize>,J:TryInto<isize>>(self,i:I,j:J)->Result<Self> where I::Error:'static+StdError,J::Error:'static+StdError{
		let (i,j)=(i.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"swap_dims",None))?,j.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"swap_dims",None))?);
		let (i,j)=(index::normalize_index(self.rank(),i).ok_or(Error::invalid_index(i,self.get_layout(),"swap_dims"))?,index::normalize_index(self.rank(),j).ok_or(Error::invalid_index(i,self.get_layout(),"swap_dims"))?);
		let mut this=self;

		if i!=j{
			this.inner.dims_mut().swap(i,j);
			this.inner.strides_mut().swap(i,j);
		}
		Ok(this)
	}
	/// convert to a owned tensor if not borrowed
	pub fn try_unique_into_tensor(self)->StdResult<Tensor<E>,Self>{
		if self.is_unique(){return Ok(self.inner)}
		Err(self)
	}
	/// attempt to conjure an axis of dim 1 from nothingness, erroring if the index would be out of bounds
	pub fn try_unsqueeze_dim<I:TryInto<isize>>(self,index:I)->Result<Self> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"squeeze",None))?;
		let index=index::normalize_range_stop(self.rank(),index).ok_or(Error::invalid_index(index,self.get_layout(),"squeeze"))?;

		let mut this=self;
		(this.inner.dims_mut().insert(index,1),this.inner.strides_mut().insert(index,1));

		Ok(this)
	}
	#[track_caller]
	/// conjure an axis of dim 1 from nothingness, erroring if the index would be out of bounds
	pub fn unsqueeze_dim(self,index:impl TryInto<isize>)->Self{self.try_unsqueeze_dim(index.try_into().unwrap_or(isize::MIN)).unwrap()}
}
impl<'a,E:'a> ViewRef<'a,E>{
	#[track_caller]
	/// clones the components of other into self, concatenating along the specified axis. may dissociate from the original tensor. no broadcasting is performed. the operation will fail if the dimensions don't match or if the resulting tensor is too big
	pub fn append(self,mut b:impl AsMut<View<E>>,i:impl TryInto<isize>)->Self where E:Clone{
		let mut this=self.into_unique();
		this.inner.append(&mut b.as_mut().0,i);

		this
	}
	/// reference as a view
	pub fn as_view(&self)->&View<E>{
		unsafe{mem::transmute(self)}
	}
	#[track_caller]
	/// apply broadcast dim to each index. errors if dims.len()!=self.rank(), or if any of the individual dims fail to broadcast
	pub fn broadcast    (&self,dims:impl AsRef<[usize]>)         ->Self{self.try_broadcast(dims).unwrap()}
	#[track_caller]
	/// broadcast dim. If the index is out of bounds of the rank, the result is an invalid index error. If dim at the index is 1 and requested size is not 1, the result is a view with the dim equal to the requested size, accomplished by index aliasing the components. If the dim at index is neither 1 nor size, and size is not 1, the result is a mismatch error. If the dim at index is size or size is 1, the result is unchanged from the input
	pub fn broadcast_dim(self,axis:impl TryInto<isize>,dim:usize)->Self{self.try_broadcast_dim(axis.try_into().unwrap_or(isize::MIN),dim).unwrap()}
	/// convert from a tensor
	pub fn from_tensor(inner:impl Into<Tensor<E>>)->Self{
		Self{inner:inner.into(),phantom:PhantomData}
	}
	/// reverse the order of components along the specified axis
	pub fn flip_dim (&self,index:impl TryInto<isize>)->Self{self.try_flip_dim(index.try_into().unwrap_or(isize::MIN)).unwrap()}
	/// reverse the order of components along all axes
	pub fn flip     (&self)->Self{
		let mut this=self.clone();

		this.inner.strides_mut().iter_mut().for_each(|s|*s=-*s);
		this
	}
	/// convert from a flat slice
	pub fn from_flat(slice:&'a [E])->Self{
		let ptr=slice.as_ptr() as *mut E;
		let len=slice.len();
		let cap=len;
												// create a new layout for a 1d tensor
		let layout=Layout::new([len]);
		unsafe{									// layout ptr gets reference incremented so it wont actually be dropped immediately. vc null because we don't have one yet
			let tensor=Tensor::from_raw_parts(&layout,ptr,len,cap,ptr::null());
			let mut ghost=tensor.clone_ref();	// increment buffer reference count. the buffer needs a +1 to vc due to being owned elsewhere
			let ghostlayout=ghost.layout_mut() as *mut Layout;
												// prevent buffer from being dropped early by forgetting the ghost buffer
			mem::forget(ghost);
			ptr::drop_in_place(ghostlayout);	// still drop the ghost layout to avoid a small memory leak

			tensor.into_view_ref()				// convert to a view ref for the appropriate lifetime
		}
	}
	/// convert from a view
	pub fn from_view  (inner:impl Into<View<E>>)  ->Self{
		Self{inner:inner.into().0,phantom:PhantomData}
	}
	/// convert into the buffer, cloning if the view's access to the buffer is not unique
	pub fn into_buffer  (self)->Vec<E> where E:Clone{
		if self.is_unique(){self.inner.into_buffer()}else{self.inner.buffer().to_vec()}
	}
	/// convert into a flattened vec
	pub fn into_flat_vec(self,mem:impl Into<Option<Vec<E>>>)->Vec<E> where E:Clone{
		if self.is_unique(){self.inner.into_flat_vec(mem)}else{self.to_flat_vec(mem)}
	}
	/// convert into a tensor, dissociating from the tensor to become an independent allocation if not unique
	pub fn into_tensor      (     self)->Tensor <   E> where E:   Clone{
		if self.is_unique(){self.inner}else{self.inner.clone()}
	}
	/// convert into a view, dissociating from the tensor to become an independent allocation if not unique
	pub fn into_view        (     self)->View   <   E> where E:   Clone{
		View(if self.is_unique(){self.inner}else{self.inner.clone()})
	}
	/// convert into a view mut, cloning if the view's access to the buffer is not unique
	pub fn into_view_mut<'b>( mut self)->ViewMut<'b,E> where E:'b+Clone{
		if self.is_borrowed(){self.inner=self.to_tensor()}
		unsafe{less_annoying_transmute(self)}
	}
	/// convert into a view ref, cloning if the view's access to the buffer is not unique
	pub fn into_view_ref<'b>( mut self)->ViewRef<'b,E> where E:'b+Clone{
		if self.is_borrowed(){self.inner=self.to_tensor()}
		unsafe{less_annoying_transmute(self)}
	}
	/// convert into a view ref
	pub fn into_unique<'b>( self)->ViewRef<'b,E> where E:'b+Clone{self.into_view_ref()}
	/// checks if the layout is valid
	pub fn is_layout_valid  (&    self)->bool{self.inner.layout().is_valid_for(false,self.inner.buffer_len())}
	#[track_caller]
	/// map the tensor components in the buffer in an unspecified order. may include components that are in the buffer but not visible to indexing, compactify first or use map indexed if this is undesirable
	pub fn map        <'b,F:FnMut(&E         )->Y,  Y:'b>(&self,mut f:F                      )->ViewRef<'b,Y>{self.try_map::        <_,Error,  Y>(|e    |Ok(f(e    ))  ).unwrap()}
	#[track_caller]
	/// map the tensor components in the buffer in an unspecified order. may include components that are in the buffer but not visible to indexing, compactify first or use map indexed if this is undesirable
	pub fn map_2      <'b,F:FnMut(&E,&X      )->Y,X,Y:'b>(&self,mut f:F,x:impl AsRef<View<X>>)->ViewRef<'b,Y>{self.try_map_2::      <_,Error,X,Y>(|x0,x1|Ok(f(x0,x1)),x).unwrap()}
	#[track_caller]
	/// map the tensor components in the buffer in order of a lex iteration over the indices. the current position is also passed to the function
	pub fn map_indexed<'b,F:FnMut(&E,Position)->Y,  Y:'b>(&self,mut f:F                      )->ViewRef<'b,Y>{self.try_map_indexed::<_,Error,  Y>(|e ,ix|Ok(f(e,ix ))  ).unwrap()}
	#[track_caller]
	/// reshape the tensor to the specified dims. errors if the component count differs. may dissociate from the original tensor
	pub fn reshape(&self,dims:impl AsRef<[usize]>)->Self where E:Clone{self.try_reshape(dims).unwrap()}
	#[track_caller]
	/// set the layout. errors if the new layout would be invalid for a shared view with this view's buffer
	pub fn set_layout(&mut self,layout:Layout){self.try_set_layout(layout).unwrap()}
	#[track_caller]
	/// slice
	pub fn slice<I:Copy+TryInto<isize>>(&self,ranges:impl AsRef<[Range<I>]>)->ViewRef<'_,E>{
		impl<T:TryInto<isize>> TryInto<isize> for TryIntoIsizeMin<T>{
			fn try_into(self)->Result<isize>{Ok(self.0.try_into().unwrap_or(isize::MIN))}
			type Error=Error;
		}

		#[derive(Clone,Copy,Debug)]
		#[repr(transparent)]
		struct TryIntoIsizeMin<T:TryInto<isize>>(T);

		let ranges:&[Range<TryIntoIsizeMin<I>>]=unsafe{mem::transmute(ranges.as_ref())};
		self.try_slice(ranges).unwrap()
	}
	#[track_caller]
	/// slice
	pub fn slice_dim<I:TryInto<isize>,J:TryInto<isize>>(&self,index:I,range:Range<J>)->ViewRef<'_,E>{self.try_slice_dim(index.try_into().unwrap_or(isize::MIN),range.start.try_into().unwrap_or(isize::MIN)..range.end.try_into().unwrap_or(isize::MIN)).unwrap()}
	#[track_caller]
	/// return an axis of dim 1 to nothingness, erroring if the dim is not 1 or the index is out of bounds
	pub fn squeeze_dim  (&self,index:impl TryInto<isize>)->Self{self.try_squeeze_dim(index.try_into().unwrap_or(isize::MIN)).unwrap()}
	#[track_caller]
	/// swap the axes
	pub fn swap_dims    (&self,i    :impl TryInto<isize>,j:impl TryInto<isize>)->Self{self.try_swap_dims(i.try_into().unwrap_or(isize::MIN),j.try_into().unwrap_or(isize::MIN)).unwrap()}
	#[track_caller]
	/// try broadcasting
	pub fn try_broadcast(&self,dims:impl AsRef<[usize]>)->Result <Self>{
		let dims=dims.as_ref();

		let (expected,rank)=(self.0.rank(),dims.len());
		if expected!=rank{
			return Err(Error::mismatch(expected,self.get_layout(),"broadcast",rank,None,"rank"));
		}
		(0..rank).try_fold(self.clone(),|acc,n|acc.try_broadcast_dim(n,dims[n])).map_err(|e|e.with_rhs_layout(Layout::from_inner(dims.to_vec(),Vec::new())))
	}
	#[track_caller]
	/// try broadcasting the specified dimension. to the size. If the index is out of bounds of the rank, the result is an invalid index error. If dim at the index is 1 and requested size is not 1, the result is a view with the dim equal to the requested size, accomplished by index aliasing the components. If the dim at index is neither 1 nor size, and size is not 1, the result is a mismatch error. If the dim at index is size or size is 1, the result is unchanged from the input
	pub fn try_broadcast_dim<I:TryInto<isize>>(&self,axis:I,dim:usize)->Result<Self> where I::Error:'static+StdError{
		let axis=axis.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"broadcast",None))?;
		let rank=self.rank();
								// try to convert the index to a usize that can index into dims
		let axis=index::normalize_index(rank,axis).ok_or(Error::invalid_index(axis,self.get_layout(),"broadcast"))?;
		let current=self.dims()[axis];
								// if the dims are already equal or the broadcast request is 1, no broadcast is needed
		if current==dim||dim==1{return Ok(self.clone())}
		if current!=1{				// if unequal and not 1, no broadcast is possible
			return Err(Error::mismatch(current,self.get_layout(),"broadcast",dim,None,"dim"))
		}
								// create a view ref to return
		let mut result=self.clone();
		let layout=result.inner.layout_mut();
								// set the dim of the view ref to the broadcasted size. stride=0 so moving along the broadcasted axis doesn't changed the referenced component
		layout.dims_mut()   [axis]=dim;
		layout.strides_mut()[axis]=0;
								// done
		Ok(result)
	}
	#[track_caller]
	/// reverse the order of components along the specified axis
	pub fn try_flip_dim     <I:TryInto<isize>>(&self,index:I         )->Result<Self> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"flip",None))?;
		let rank=self.rank();
								// try to convert the index to a usize that can index into dims
		let index=index::normalize_index(rank,index).ok_or(Error::invalid_index(index,self.get_layout(),"flip"))?;
		let mut this=self.clone();

		this.inner.strides_mut()[index]*=-1;
		Ok(this)
	}
	#[track_caller]
	/// map the tensor components in the buffer in an unspecified order. may include components that are in the buffer but not visible to indexing, compactify first or use map indexed if this is undesirable
	pub fn try_map<'b,F:FnMut(&E)->StdResult<Y,R>,R:'static+StdError,Y:'b>(&self,mut f:F)->Result<ViewRef<'b,Y>>{
		let mut buffer=Vec::with_capacity(self.layout().len());
		for x in self.buffer(){buffer.push(f(x).map_err(|e|Error::other(e,self.get_layout(),"map",None))?)}

		let layout=self.get_layout();
		Ok(ViewRef{inner:Tensor::from_inner(buffer,layout),phantom:PhantomData})
	}
	#[track_caller]
	/// map the tensor components in the buffer in an unspecified order. may include components that are in the buffer but not visible to indexing, compactify first or use map indexed if this is undesirable
	pub fn try_map_2<'b,F:FnMut(&E,&X)->StdResult<Y,R>,R:'static+StdError,X,Y:'b>(&self,mut f:F,x:impl AsRef<View<X>>)->Result<ViewRef<'b,Y>>{
		let (ll,rl)=(self.get_layout(),x.as_ref().get_layout());
		let x0=self;
		let x1=x.as_ref();
		// TODO make mutual broadcast function that also extends rank
		let (x0,x1)=(x0.view_ref().try_broadcast(rl.dims())?,x1.view_ref().try_broadcast(ll.dims())?);
		x0.try_map_indexed(|x,ix|f(x,&x1[ix])).map_err(|e|e.with_rhs_layout(rl))
	}
	#[track_caller]
	/// map the tensor components in the buffer in order of a lex iteration over the indices. the current position is also passed to the function
	pub fn try_map_indexed<'b,F:FnMut(&E,Position)->StdResult<Y,R>,R:'static+StdError,Y:'b>(&self,mut f:F)->Result<ViewRef<'b,Y>>{
		let mut buffer=Vec::with_capacity(self.count());
		for ix in self.indices(){buffer.push(f(&self[&ix],ix).map_err(|e|Error::other(e,self.get_layout(),"map",None))?)}

		let layout=Layout::new(self.dims());
		Ok(ViewRef{inner:Tensor::from_inner(buffer,layout),phantom:PhantomData})
	}
	#[track_caller]
	/// attempt to set the layout. errors if the new layout would be invalid for a shared view with this view's buffer
	pub fn try_set_layout   (&mut self,layout:Layout)->Result<()>{
		layout.check_validity_for(false,self.buffer_len()).map_err(|e|e.with_op("set layout"))?;
		Ok(*self.inner.layout_mut()=layout)
	}
	#[track_caller]
	/// attempt to reshape the tensor to the specified dims. errors if the component count differs. may dissociate from the original tensor
	pub fn try_reshape(&self,dims:impl AsRef<[usize]>)->Result<Self> where E:Clone{
		let dims=dims.as_ref();
		let newcount=dims.iter().try_fold(1,|acc:usize,&item|acc.checked_mul(item)).ok_or_else(||Error::too_big(Some(self.buffer_len()),None,self.get_layout(),Some(self.layout().len()),"reshape"))?;

		if self.count()!=newcount{return Err(Error::mismatch(self.count(),self.get_layout(),"reshape",newcount,Layout::new(dims),"count"))}

		let mut this=self.clone();
		if !this.is_layout_normalized(){
			this=this.into_unique();
			this.inner.normalize_layout();
		}

		*this.inner.layout_mut()=Layout::new(dims);
		Ok(this)
	}
	#[track_caller]
	/// slice
	pub fn try_slice    <            J:Copy+TryInto<isize>>(&self,ranges:impl AsRef<[Range<J>]>)->Result<ViewRef<'_,E>> where                           J::Error:'static+StdError{
		let ranges=ranges.as_ref();
		let rank  =self.rank();
											// check rank
		if ranges.len()!=rank{return Err(Error::mismatch(rank,self.get_layout(),"slice",ranges.len(),None,"rank"))}
		let mut result=self.clone();

		for n in 0..rank{
			let probe=result.try_slice_dim(n,ranges[n].clone())?;
			unsafe{
				let layout=probe.get_layout();
				let offset=probe.as_ptr().offset_from(result.as_ptr()) as usize;

				mem::drop(probe);

				*result.inner.layout_mut()=layout;
				result.inner.offset(offset);
			}
		}

		Ok(result)
	}
	#[track_caller]
	/// slice
	pub fn try_slice_dim<I:TryInto<isize>,J:TryInto<isize>>(&self,index:I,slicerange:Range<J>  )->Result<ViewRef<'_,E>> where I::Error:'static+StdError,J::Error:'static+StdError{
		let index=index           .try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"slice",None))?;
		let rank =self.rank();
		let start=slicerange.start.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"slice",None))?;
		let stop =slicerange.end  .try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"slice",None))?;

		let index=index::normalize_index     (rank,index).ok_or(Error::invalid_index(index,self.get_layout(),"slice"))?;
		let dim=self.dims()[index];

		let start=index::normalize_index     (dim ,start).ok_or(Error::invalid_index(start,self.get_layout(),"slice"))?;
		let stop =index::normalize_range_stop(dim ,stop ).ok_or(Error::invalid_index(stop, self.get_layout(),"slice"))?;

		let mut this=self.clone();
		let rev=stop<start;
		let start=start.min(stop);
		let stop =start.max(stop);

		let (dims,strides)=this.inner.layout_mut().inner_mut();
		let offset=if strides[index]<0{dim-stop}else{start}*strides[index].abs() as usize;

		dims[index]=stop-start;
		if rev{strides[index]*=-1}

		unsafe{			// buffer isn't dropped by this because self outlive its
			this.inner.offset(offset);
		}
		Ok(this)
	}
	#[track_caller]
	/// attempt to return an axis of dim 1 to nothingness, erroring if the dim is not 1 or the index is out of bounds
	pub fn try_squeeze_dim<I:TryInto<isize>>(&self,index       :I    )->Result<Self> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"squeeze",None))?;
		let index=index::normalize_index(self.rank(),index).ok_or(Error::invalid_index(index,self.get_layout(),"squeeze"))?;

		let dim=self.dims()[index];
		if dim!=1{return Err(Error::mismatch(dim,self.get_layout(),"squeeze",1,None,"dim"))}

		let mut this=self.clone();
		(this.inner.dims_mut().remove(index),this.inner.strides_mut().remove(index));

		Ok(this)
	}
	#[track_caller]
	/// swap the axes if in bounds
	pub fn try_swap_dims  <I:TryInto<isize>,J:TryInto<isize>>(&self,i:I,j:J)->Result<Self> where I::Error:'static+StdError,J::Error:'static+StdError{
		let (i,j)=(i.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"swap_dims",None))?,j.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"swap_dims",None))?);
		let (i,j)=(index::normalize_index(self.rank(),i).ok_or(Error::invalid_index(i,self.get_layout(),"swap_dims"))?,index::normalize_index(self.rank(),j).ok_or(Error::invalid_index(j,self.get_layout(),"swap_dims"))?);
		let mut this=self.clone();

		if i!=j{
			this.inner.dims_mut().swap(i,j);
			this.inner.strides_mut().swap(i,j);
		}
		Ok(this)
	}
	/// convert to a owned tensor if not borrowed
	pub fn try_unique_into_tensor(self)->StdResult<Tensor<E>,Self>{
		if self.is_unique(){return Ok(self.inner)}
		Err(self)
	}
	#[track_caller]
	/// attempt to conjure an axis of dim 1 from nothingness, erroring if the index would be out of bounds
	pub fn try_unsqueeze_dim<I:TryInto<isize>>(&self,index:I)->Result<Self> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(Box::new(e),self.get_layout(),"squeeze",None))?;
		let index=index::normalize_range_stop(self.rank(),index).ok_or(Error::invalid_index(index,self.get_layout(),"squeeze"))?;

		let mut this=self.clone();
		(this.inner.dims_mut().insert(index,1),this.inner.strides_mut().insert(index,1));

		Ok(this)
	}
	#[track_caller]
	/// conjure an axis of dim 1 from nothingness, erroring if the index would be out of bounds
	pub fn unsqueeze_dim(&self,index:impl TryInto<isize>)->Self{self.try_unsqueeze_dim(index.try_into().unwrap_or(isize::MIN)).unwrap()}
}

unsafe fn less_annoying_transmute<T,U>(input:T)->U{// TODO this should be refactored out when I'm less annoyed
	let output=unsafe{mem::transmute_copy(&input)};
	mem::forget(input);

	output
}
#[derive(Debug,Default)]
#[repr(transparent)]
/// tensor view type allowing flipping, slicing, some reshaping, transposition, without reallocating. tensor uses in place operations by default. view tends to use moving operations; viewmut/viewref tend to use by reference operations. layout validity: view and tensor allow invalid layouts but behaves incorrectly or panics when used for anything, viewmut and viewref have guaranteed layout validity.
pub struct View<E>(Tensor<E>);	// vc safety: any >1 if ref transmuted, 1 if owned
#[derive(Debug)]
#[repr(transparent)]
/// wraps a mutable view type allowing flipping, slicing, some reshaping, transposition, without necessarily reallocating. tensor uses in place operations by default. view tends to use moving operations; viewmut/viewref tend to use by reference operations. Although ViewMut is Clone, cloning it will actually clone the inner data and dissociate from the original tensor. Methods that may dissociate from the original tensor should be documented as such. Mutation to components will modify the original tensor if and only if not dissociated. layout validity: view and tensor allow invalid layouts but behaves incorrectly or panics when used for anything, viewmut and viewref have guaranteed layout validity.
pub struct ViewMut<'a,E:'a>{
	inner:Tensor<E>,			// vc safety: at least the number of view to the tensor, +1 if the tensor is borrowed
	phantom:PhantomData<&'a mut E>
}
#[derive(Debug)]
#[repr(transparent)]
/// wraps a shared view type allowing flipping, slicing, some reshaping, transposition, without necessarily reallocating. tensor uses in place operations by default. view tends to use moving operations; viewref tend to use by reference operations. Methods that may dissociate from the original tensor should be documented as such. layout validity: view and tensor allow invalid layouts but behaves incorrectly or panics when used for anything, viewmut and viewref have guaranteed layout validity
pub struct ViewRef<'a,E:'a>{
	inner:Tensor<E>,			// vc safety: at least the number of view to the tensor, +1 if the tensor is borrowed
	phantom:PhantomData<&'a E>,
}

use std::{
	borrow::{Borrow,BorrowMut,ToOwned},cmp::{Eq,PartialEq},error::Error as StdError,hash::{Hash,Hasher},iter::FromIterator,marker::PhantomData,mem,ops::{Deref,DerefMut,Index,IndexMut,Range},ptr,slice,result::Result as StdResult
};
use super::{Error,GridIter,Layout,Position,Result,Tensor,index};
