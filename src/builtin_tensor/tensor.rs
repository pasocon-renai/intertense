impl<E:Add<X,Output=Y>+Clone,X:Clone,Y> Add<Tensor<X>> for Tensor<E>{
	fn add(self,rhs:Tensor<X>)->Self::Output{self.into_view_ref().map_2(|x0,x1|x0.clone()+x1.clone(),rhs).try_unique_into_tensor().ok().unwrap()}
	type Output=Tensor<Y>;
}
impl<E> AsMut    <Self   >   for Tensor<E>{
	fn as_mut(&mut self)->&mut Self{self}
}
impl<E> AsMut    <View<E>>   for Tensor<E>{
	fn as_mut(&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E> AsRef    <Self   >   for Tensor<E>{
	fn as_ref(&    self)->&Self{self}
}
impl<E> AsRef    <View<E>>   for Tensor<E>{
	fn as_ref    (&    self)->&    View<E>{self.as_view()}
}
impl<E> Borrow   <View<E>>   for Tensor<E>{
	fn borrow    (&    self)->&    View<E>{self.as_view()}
}
impl<E> BorrowMut<View<E>>   for Tensor<E>{
	fn borrow_mut(&mut self)->&mut View<E>{self.as_mut_view()}
}
impl<E:Clone> Clone          for Tensor<E>{
	fn clone(&self)->Self{self.to_tensor()}
	/*fn clone_from(&mut self,other:&Self){
		self.as_mut_buffer().clone_from(other.as_buffer());
		self.layout_mut   ().clone_from(other.layout   ());
	}*/
}
impl<E:Debug> Debug          for Tensor<E>{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{(self.layout(),self.buffer()).fmt(f)}
}
impl<E> Default              for Tensor<E>{
	fn default()   ->Self{Self::new(Vec::new(),Vec::new())}
}
impl<E:Eq> Eq                for Tensor<E>{}
impl<E> Deref                for Tensor<E>{
	fn deref    (&    self)->&    Self::Target{
		unsafe{mem::transmute(self)}		// view is a transparent repr of tensor, view is not owned so no concern about the counter as lifetime rules will prevent contradictory access
	}
	type Target=View<E>;
}
impl<E> DerefMut             for Tensor<E>{
	fn deref_mut(&mut self)->&mut Self::Target{
		unsafe{mem::transmute(self)}		// view is a transparent repr of tensor, view is not owned so no concern about the counter as lifetime rules will prevent contradictory access
	}
}
impl<E> Drop                 for Tensor<E>{
	fn drop     (&mut self){
		unsafe{		// decrement and release the view count. if it's not 1 it's not time to drop yet
			if Arc::strong_count(&self.viwcnt)>1{return}
					// drop the buffer from an acquired fence
			atomic::fence(Acquire);
			mem::drop(Vec::from_raw_parts(self.ptr,self.len,self.cap))
		}
	}
}
impl<E:Clone        > From<       &[E  ] > for Tensor<E>{
	fn from(inner:&[E ])->Self{Self::from_flat(inner)}
}
impl<E,const N:usize> From<        [E;N] > for Tensor<E>{
	fn from(inner:[E;N])->Self{Self::from_vec(inner)}
}
impl<   E:   Clone  > From<&Tensor< E   >> for Tensor<E>{// TODO from view ref
	fn from(inner:&Tensor  <   E>)->Self{inner.clone().into_tensor()}
}
impl<   E           > From< Vec   < E   >> for Tensor<E>{
	fn from(inner:  Vec    <   E>)->Self{Self::from_vec(inner)}
}
impl<   E           > From< View  < E   >> for Tensor<E>{
	fn from(inner:  View   <   E>)->Self{inner        .into_tensor()}
}
impl<   E:   Clone  > From<&View  < E   >> for Tensor<E>{
	fn from(inner: &View   <   E>)->Self{inner.to_owned().into_tensor()}
}
impl<E> FromIterator<E> for Tensor<E>{
	fn from_iter<I:IntoIterator<Item=E>>(iter:I)->Self{Self::from_vec(iter)}
}
impl<E:Hash> Hash            for Tensor<E>{
	fn hash<H:Hasher>(&self,hasher:&mut H){
		for ix in self.indices(){self[ix].hash(hasher)}
		self.dims().hash(hasher);
	}
}
impl<E> Index< Position>     for Tensor<E>{
	fn index(&self,index: Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E> Index<&Position>     for Tensor<E>{
	fn index(&self,index:&Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<E,I:     TryInto<isize>,const N:usize> Index< [I;N]> for Tensor<E>{
	fn index(&self,index: [I;N])->&Self::Output{self.index(index.map(|x|x.try_into().unwrap_or(isize::MIN)).as_slice())}
	type Output=E;
}
impl<E,I:Copy+TryInto<isize>              > Index<&[I]  > for Tensor<E>{
	fn index(&self,index:&[I])  ->&Self::Output{
		unsafe{std_slice::from_raw_parts(self.ptr,self.len)}.index(index::compute_offset(self.dims(),index,0,self.strides()).unwrap())
	}
	type Output=E;
}
impl<E> IndexMut< Position>  for Tensor<E>{
	fn index_mut(&mut self,index: Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E> IndexMut<&Position>  for Tensor<E>{
	fn index_mut(&mut self,index:&Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<E,I:     TryInto<isize>,const N:usize> IndexMut< [I;N]> for Tensor<E>{
	fn index_mut(&mut self,index: [I;N]   )->&mut Self::Output{self.index_mut(index.map(|x|x.try_into().unwrap_or(isize::MIN)).as_slice())}
}
impl<E,I:Copy+TryInto<isize>              > IndexMut<&[I]  > for Tensor<E>{
	fn index_mut(&mut self,index:&[I]     )->&mut Self::Output{
		unsafe{std_slice::from_raw_parts_mut(self.ptr,self.len)}.index_mut(index::compute_offset(self.dims(),index,0,self.strides()).unwrap())
	}
}
impl<E:PartialEq<X>,X>                      PartialEq<Tensor<X>> for Tensor<E>{
	fn eq(&self,other:&Tensor<X>)->bool{self.dims()==other.dims()&&self.indices().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&Tensor<X>)->bool{self.dims()!=other.dims()||self.indices().any(|ix|self[&ix]!=other[&ix])}
}
impl<E:PartialEq<X>,X>                      PartialEq<View<X  >> for Tensor<E>{
	fn eq(&self,other:&  View<X>)->bool{self.dims()==other.dims()&&self.indices().all(|ix|self[&ix]==other[&ix])}
	fn ne(&self,other:&  View<X>)->bool{self.dims()!=other.dims()||self.indices().any(|ix|self[&ix]!=other[&ix])}
}
impl<E:RefUnwindSafe> RefUnwindSafe for Tensor<E>{}
unsafe impl<E:Send  > Send          for Tensor<E>{}		// impl send sync based on E, like Vec, since the not sendsync inside is just a destructured vec
unsafe impl<E:Sync  > Sync          for Tensor<E>{}

impl<E> Tensor<E>{
	#[track_caller]
	/// moves the components of other into self, concatenating along the specified axis. no broadcasting is performed. the operation will fail if the dimensions don't match or if the resulting tensor is too big
	pub fn append       (&mut self,b:&mut Tensor<E>,i:impl TryInto<isize>)->&mut Self{self.try_append(b,i.try_into().unwrap_or(isize::MIN)).unwrap()}
	/// get a pointer to the data buffer
	pub fn as_mut_ptr   (&mut self)->*mut   E{self.ptr as *mut   E}
	/// convert to a mutable view reference
	pub fn as_mut_view  (&mut self)->&mut View<E>{
		unsafe{mem::transmute(self)}	// view is a transparent repr of tensor, view is not owned so no concern about the counter as lifetime rules will prevent contradictory access
	}
	/// get a pointer to the data buffer
	pub fn as_ptr       (&    self)->*const E{self.ptr as *const E}
	/// convert to a view reference
	pub fn as_view      (&    self)->&View<E>{
		unsafe{mem::transmute(self)}	// view is a transparent repr of tensor, view is not owned so no concern about the counter as lifetime rules will prevent contradictory access
	}
	#[track_caller]
	/// convert to a viewmut reference
	pub fn as_view_mut  (&mut self)->&mut ViewMut<'_,E>{
		self.layout.check_validity_for(true, self.len).unwrap();
		unsafe{mem::transmute(self)}	// transparent ref, since the ViewMut is just the tensor, for vc purposes it's not an additional view to the same buffer any more than a mutable reborrow is an additional mutable reference to the same data.
	}
	#[track_caller]
	/// convert to a viewref reference
	pub fn as_view_ref  (&    self)->&    ViewRef<'_,E>{
		self.layout.check_validity_for(false,self.len).unwrap();
		unsafe{mem::transmute(self)}	// transparent ref, since the ViewRef is just the tensor, for vc purposes it's not an additional view to the same buffer any more than a mutable reborrow is an additional mutable reference to the same data.
	}
	#[track_caller]
	/// apply broadcast dim to each index. errors if dims.len()!=self.rank(), or if any of the individual dims fail to broadcast
	pub fn broadcast    (&mut self,dims:impl AsRef<[usize]>)          ->&mut Self where E:Clone{
		*self=mem::take(self).into_view().broadcast(dims).into_tensor();
		self
	}
	#[track_caller]
	/// broadcast dim. If the index is out of bounds of the rank, the result is an invalid index error. If dim at the index is 1 and requested size is not 1, the result is a view with the dim equal to the requested size, accomplished by index aliasing the components. If the dim at index is neither 1 nor size, and size is not 1, the result is a mismatch error. If the dim at index is size or size is 1, the result is unchanged from the input
	pub fn broadcast_dim(&mut self,axis:impl TryInto<isize>,dim:usize)->&mut Self where E:Clone{
		*self=mem::take(self).into_view().broadcast_dim(axis,dim).into_tensor();
		self
	}
	/// reference the underlying buffer
	pub fn buffer       (&    self)->&[E]{
		unsafe{std_slice::from_raw_parts    (self.ptr as *const E,self.len)}
	}
	/// get the capacity of the underlying buffer
	pub fn buffer_cap   (&    self)->usize{self.cap}
	/// get the length of the underlying buffer
	pub fn buffer_len   (&    self)->usize{self.len}
	/// reference the underlying buffer
	pub fn buffer_mut   (&mut self)->&mut [E]{
		unsafe{std_slice::from_raw_parts_mut(self.ptr as *mut   E,self.len)}
	}
	#[track_caller]
	/// concatenate a collection of tensors along the specified axis
	pub fn cat<I:IntoIterator>(collection:I,index:impl TryInto<isize>)->Self where I::Item:Into<Self>{Self::try_cat(collection,index.try_into().unwrap_or(isize::MIN)).unwrap()}
	#[track_caller]
	/// check if append would succeed
	pub fn check_append<I:TryInto<isize>>(&    self,b:&Tensor<E>,index:I)->Result<()> where I::Error:'static+StdError{
		let (layout,otherlayout)=(&self.layout,&b.layout);
		let index=index.try_into().map_err(|e|Error::other(e,self.get_layout(),"append",None))?;
		let rank=layout.rank();
									// check for errors
		let index=match index::normalize_index(rank,index).ok_or(Error::invalid_index(index,layout.clone(),"append")){Ok(d)=>d,Err(e)=>return Err(e)};
		for n in 0..rank{
			if index==n{continue}
			if layout.dims()[n]!=otherlayout.dims()[n]{
				return Err(Error::mismatch(otherlayout.dims()[n],layout.clone(),"append",otherlayout.dims()[n],otherlayout.clone(),"dim"))
			}
		}
		Ok(())
	}
	#[track_caller]
	/// check if broadcast would succeed
	pub fn check_broadcast(&self,dims:&[usize])->Result<()>{self.view_ref().try_broadcast(dims).map(|_|())}
	#[track_caller]
	/// check if broadcast would succeed
	pub fn check_broadcast_dim<I:TryInto<isize>>(&self,axis:  isize,dim  :usize)->Result<()> where I::Error:'static+StdError{self.view_ref().try_broadcast_dim(axis,dim ).map(|_|())}
	#[track_caller]
	/// check if flip dim would succeed
	pub fn check_flip_dim     <I:TryInto<isize>>(&self,index:I)                 ->Result<()> where I::Error:'static+StdError{self.view_ref().try_flip_dim(index).map(|_|())}
	/// create another handle to the tensor for use in a ViewRef. To prevent mutable aliasing, neither self nor the result should have their buffer mutated while both are alive. layout manipulation is acceptable.
	pub (crate) unsafe fn clone_ref(&self)->Self{
		Self{
			layout:self.layout.clone(),
			viwcnt:self.viwcnt.clone(),
			ptr:   self.as_ptr() as *mut E,
			cap:   self.buffer_cap(),len:self.buffer_len()
		}
	}
	#[track_caller]
	/// normalizes the layout and removes excess capacity
	pub fn compactify(&mut self){
		self.normalize_layout();
		let (mut buffer,layout)=mem::take(self).into_inner();

		buffer.truncate(layout.count());
		buffer.shrink_to_fit();
		*self=Self::from_inner(buffer,layout);
	}
	/// return the number of components in the tensor
	pub fn count        (&    self)->         usize {self.layout.count()}
	/// reference the dims
	pub fn dims         (&    self)->&    Vec<usize>{self.layout.dims()}
	/// reference the dims
	pub fn dims_mut     (&mut self)->&mut Vec<usize>{self.layout.dims_mut()}
	#[track_caller]
	/// reverse the order of components along the specified axis
	pub fn flip_dim     (&mut self,index:impl TryInto<isize>)->&mut Self{
		self.layout=self.view_ref().flip_dim(index).get_layout();
		self
	}
	/// reverse the order of components along all axes
	pub fn flip         (&mut self)->&mut Self{
		self.strides_mut().iter_mut().for_each(|s|*s=-*s);
		self
	}
	//#[track_caller]
	// flatten the tensor into 1 dim // TODO flatten shape flatten type
	//pub fn flatten_shape(&mut self){*self=mem::replace(self,Self::new(Vec::new(),Vec::new())).into_view_mut().flatten_shape(index,size)}
	// reverse the direction of all axes // TODO flip dim
	//pub fn flip         (&mut self){*self=mem::replace(self,Self::new(Vec::new(),Vec::new())).into_view_mut().flip()}
	/// create a new tensor from a flat slice
	pub fn from_flat  (flat  :impl AsRef<[E]>)->Self where E:Clone{Self::from_vec(flat.as_ref().to_vec())}
	/// create a new tensor from a buffer and layout
	pub fn from_inner (buffer:impl Into<Vec<E>>,layout:impl Into<Layout>)->Self{
		let (mut buffer,layout)=(buffer.into(),layout.into());
		let (ptr,len,cap)=(buffer.as_mut_ptr(),buffer.len(),buffer.capacity());

		mem::forget(buffer);
		Self{
			layout,ptr,len,cap,
			viwcnt:().into()
		}
	}
	/// creates tensor from raw parts:
	/// capacity: capacity of the underlying buffer
	/// ptr: pointer to the buffer
	/// layout: valid pointer to a layout. layouts are reference counted so it doesn't need to have any specific lifetime; the reference count will be incremented
	/// len: length of the underlying buffer
	/// vc: some extra view related information for managing reference wrappings. may be null. probably just a counter but subject to change
	pub unsafe fn from_raw_parts(layout:*const Layout,ptr:*mut E,len:usize,cap:usize,vc:*const ())->Self{
		Self{
			layout:unsafe{&*layout}.clone(),
			viwcnt:unsafe{
				if vc.is_null(){().into()}else{Arc::from_raw(vc)}
			},
			ptr,len,cap
		}
	}
	/// create a new tensor from another tensor
	pub fn from_tensor(view  :impl Into<View<E>>)->Self{
		unsafe{mem::transmute(view.into())}
	}
	/// create a new tensor from a collection
	pub fn from_vec   (vec   :impl IntoIterator<Item=E>)->Self{
		let data:Vec<E>=vec.into_iter().collect();
		let dims=vec![data.len()];

		Self::new(data,dims)
	}
	/// create a new tensor from a view
	pub fn from_view  (view  :impl Into<View<E>>)->Self{
		unsafe{mem::transmute(view.into())}
	}
	/// creates a new tensor with the given dimensions, full of the same value
	pub fn full<D:AsRef<[usize]>>(data:E,dims:D)->Self where E:Clone{Self::new_with(||data.clone(),Layout::new(dims))}
	/// get the layout
	pub fn get_layout(&self)->Layout{self.layout.clone()}
	/// returns an iterator over the view indices
	pub fn indices(&self)->GridIter{GridIter::from_shared_layout(self.layout.clone())}
	#[track_caller]
	/// flattens the tensor into a vec
	pub fn into_flat_vec(self,mem:impl Into<Option<Vec<E>>>)->Vec<E>{
		let mut mem=mem.into().unwrap_or_default();
		if self.is_layout_normalized(){
			let count=self.dims()[0]*self.strides()[0].abs() as usize;
			let mut buffer=self.into_buffer();

			buffer.truncate(count);
			if mem.capacity()<count&&mem.len()==0{mem=buffer}else{mem.extend(buffer)}

			mem
		}else{
			let count =self.count();
			let layout=self.get_layout();
			let mut position=vec![0;self.rank()];

			mem.reserve(count);

			let mut data:Vec<Option<E>>=self.into_buffer().into_iter().map(Some).collect();
			let dims   =layout.dims   ();
			let strides=layout.strides();

			index::for_positions(dims,|ix|mem.push(data[index::compute_offset(dims,ix,0,strides).unwrap()].take().unwrap()),&mut position);
			mem
		}
	}
	/// convert into the inner buffer
	pub fn into_buffer(self)-> Vec<E>{self.into_inner().0}
	/// convert into the buffer and layout
	pub fn into_inner (self)->(Vec<E>,Layout){// TODO seg drop
		unsafe{		// reconstruct the buffer knowing it won't alias ownership. The buffer being valid as a vec is an invariant of this struct
			if self.view_count()!=1{panic!("internal error: exposed tensor should only be possible to move when view count is 1")}
			atomic::fence(Acquire);

			let r=(Vec::from_raw_parts(self.ptr,self.len,self.cap),self.layout.clone());
			mem::forget(self);
			r
		}
	}
	/// convert into a tensor
	pub fn into_tensor(self)->Tensor <   E>{
		unsafe{mem::transmute(self)}
	}
	/// convert a tensor into a view
	pub fn into_view  (self)->View   <   E>{
		unsafe{mem::transmute(self)}
	}
	/// convert a tensor into a view
	pub fn into_view_mut<'a>   ( self)->ViewMut<'a,E> where E:'a{
		self.layout.check_validity_for(false,self.len).unwrap();
		unsafe{mem::transmute(self)}
	}
	/// convert a tensor into a view
	pub fn into_view_ref<'a>   ( self)->ViewRef<'a,E> where E:'a{
		self.layout.check_validity_for(true ,self.len).unwrap();
		unsafe{mem::transmute(self)}
	}
	/// checks if the tensor is empty
	pub fn is_empty            (&self)->bool{self.dims().contains(&0)}
	/// checks if the layout is contiguous with lex ordered indices and positive strides, returning true if it is and false if it isn't
	pub fn is_layout_normalized(&self)->bool{self.layout.is_normalized()}
	/// checks if the layout is valid
	pub fn is_layout_valid     (&self)->bool{self.layout().check_validity_for(true,self.len).is_ok()}
	/// reference the layout
	pub fn layout    (&    self)->&    Layout{&    self.layout}
	/// reference the layout
	pub fn layout_mut(&mut self)->&mut Layout{&mut self.layout}
	/// apply a function to every component in an unspecified order, mapping them in place
	pub fn map<F:FnMut(E)->E>(&mut self,mut f:F){
		//if self.strides().iter().copied().min().;

		let (mut buffer,layout)=mem::take(self).into_inner();	// destructure the tensor
		buffer.truncate(layout.len());							// trim excess components if present.

		buffer=buffer.into_iter().map(|x|f(x)).collect();		// stdlib vec map optimization go brr
		*self=Self::from_inner(buffer,layout);					// reconstruct
	}


	/// apply a function to every component. panics if the dims mismatched. this doesn't broadcast, use the view version of the function that works by reference for automatic broadcasting
	pub fn map_2<F:FnMut(E,X)->Y,X,Y>(mut self,mut f:F,mut x:Tensor<X>)->Tensor<Y>{
		assert_eq!(self.dims(),x.dims());
		(self.normalize_layout(),x.normalize_layout());

		let (buffer,layout)=self.into_inner();
		let count=layout.dims()[0]*layout.strides()[0].abs() as usize;

		Tensor::new(buffer.into_iter().zip(x.into_buffer()).map(|(e,x)|f(e,x)).take(count).collect(),layout.into_dims())
	}
	/// apply a function to every component, in order, along with the component indices
	pub fn map_indexed<F:FnMut(E,Position)->E>(&mut self,mut f:F){
		self.normalize_layout();

		let mut indices=self.indices();
		self.map(|x|f(x,indices.next().unwrap()));
	}
	/// creates a new tensor from the data and dimensions
	pub fn new(data:Vec<E>,dims:Vec<usize>)->Self{
		Self::from_inner(data,Layout::new(dims))
	}
	/// creates a new tensor from the dimensions, and default values for the components
	pub fn new_with<F:FnMut()->E>(datagen:F,layout:impl Into<Layout>)->Self{
		let mut data=Vec::new();
		let layout=layout.into();

		data.resize_with(layout.len(),datagen);
		Self::from_inner(data,layout)
	}
	/// creates a new tensor from the data and dimensions. layout is not checked, but attempting operations with an invalid layout will likely result in incorrect behavior or panics
	pub fn new_with_layout(data:Vec<E>,layout:Layout)->Self{Self::from_inner(data,layout)}
	/// rearranges the components in memory and layout so the layout is normalized. may panic or behave unexpectedly if the layout is invalid
	pub fn normalize_layout(&mut self){
		if self.is_layout_normalized(){return}

		let layout=self.layout.clone();
		let buffer=mem::take(self).into_flat_vec(None);

		*self=Self::new(buffer,layout.into_dims())
	}
	/// offset the buffer. offset must not be more than the remaining length of the buffer. do not use on a tensor that may drop the buffer on drop
	pub (crate) unsafe fn offset(&mut self,offset:usize){
		unsafe{self.ptr=self.ptr.add(offset)}
	}
	/// pads the dimension to size with val.
	pub fn pad_dim(&mut self,dim:isize,size:usize,val:E) where E:Clone{self.pad_dim_with(dim,size,||val.clone())}
	/// pads the dimension to size with val.
	pub fn pad_dim_with<F:FnMut()->E>(&mut self,dim:isize,size:usize,val:F){
		let dim=if let Some(d)=index::normalize_index(self.dims().len(),dim){d}else{panic!()} as isize;
		if self.dims()[dim as usize]>=size{return}

		self.swap_dims(dim,0);
		self.normalize_layout();

		let (mut buffer,mut layout)=mem::take(self).into_inner();

		buffer.resize_with(size*layout.strides()[0].abs() as usize,val);
		layout.dims_mut()[0]=size;

		*self=Self::from_inner(buffer,layout);

		self.swap_dims(dim,0);
	}
	/// mtaches the dims of same rank tensors by padding
	pub fn pad_match<'a,A:IntoIterator<Item=&'a mut I>,I:'a+AsMut<Tensor<E>>>(tensors:A,val:E) where E:Clone{Self::pad_match_with(tensors,||val.clone())}
	/// mtaches the dims of same rank tensors by padding
	pub fn pad_match_with<'a,A:IntoIterator<Item=&'a mut I>,F:FnMut()->E,I:'a+AsMut<Tensor<E>>>(tensors:A,mut val:F){
		let mut tensors:Vec<&'a mut I>=tensors.into_iter().collect();
		if tensors.len()==0{return}

		let mut dims=tensors[0].as_mut().dims().to_vec();

		for d in tensors.iter_mut().map(|i|i.as_mut().dims()){
			if d.len()!=dims.len(){panic!()}
			for (&d,e) in d.iter().zip(dims.iter_mut()){*e=d.max(*e)}
		}
		for t in tensors.iter_mut().map(|i|i.as_mut()){
			for d in 0..dims.len(){
				t.pad_dim_with(d as isize,dims[d],&mut val)
			}
		}
	}
	/// get the tensor rank
	pub fn rank       (&    self)->         usize {self.layout.rank   ()}
	#[track_caller]
	/// reshape
	pub fn reshape(&mut self,shape:impl AsRef<[usize]>)->&mut Self where E:Clone{// TODO this technically doesn't need clone
		*self=mem::take(self).into_view_mut().reshape(shape).try_unique_into_tensor().ok().unwrap();
		 self
	}
	#[track_caller]
	/// sets the layout, panicing if invalid
	pub fn set_layout(&mut self,layout:Layout){
		let mut v=mem::take(self).into_view_mut();
		v.set_layout(layout);

		*self=v.try_unique_into_tensor().ok().unwrap();
	}#[track_caller]
	/// slice
	pub fn slice<I:Copy+TryInto<isize>>(&mut self,ranges:impl AsRef<[Range<I>]>)->&mut Self{
		*self=mem::take(self).into_view().slice(ranges).into_tensor();
		 self
	}
	#[track_caller]
	/// slice
	pub fn slice_dim<I:TryInto<isize>,J:TryInto<isize>>(&mut self,index:I,range:Range<J>)->&mut Self{
		*self=mem::take(self).into_view().slice_dim(index,range).into_tensor();
		 self
	}
	#[track_caller]
	/// remove a dim if it's size 1
	pub fn squeeze_dim(&mut self,dim:impl TryInto<isize>)->&mut Self{
		self.layout=self.view_ref().squeeze_dim(dim).get_layout();
		self
	}
	/// swap dims
	pub fn swap_dims(&mut self,a:impl TryInto<isize>,b:impl TryInto<isize>)->&mut Self{
		self.layout=self.view_ref().swap_dims(a,b).get_layout();
		self
	}
	#[track_caller]
	/// stack a collection of tensors along an axis inserted at the specified index
	pub fn stack<I:IntoIterator>(collection:I,index:impl TryInto<isize>)->Self where I::Item:Into<Self>{Self::try_stack(collection,index.try_into().unwrap_or(isize::MIN)).unwrap()}

	/// reference the strides
	pub fn strides    (&    self)->&    Vec<isize>{self.layout.strides()}
	/// reference the strides
	pub fn strides_mut(&mut self)->&mut Vec<isize>{self.layout.strides_mut()}
	/// transmute the buffer type for erasure shenanigans
	pub (crate) unsafe fn transmute_components<X>(mut self)->Tensor<X>{
		let mut stand:Tensor<X>=Tensor::default();

		mem::swap(&mut self.layout,&mut stand.layout);
		mem::swap(&mut self.viwcnt,&mut stand.viwcnt);
		mem::swap(&mut self.len,&mut stand.len);
		mem::swap(&mut self.cap,&mut stand.cap);

		stand.ptr=self.ptr as *mut X;
		self.ptr=ptr::dangling_mut();

		stand
	}
	/// moves the components of other into self, concatenating along the specified axis. no broadcasting is performed. the operation will fail if the dimensions don't match or if the resulting tensor is too big. may panic rather than return err if the layout is invalid
	pub fn try_append<I:TryInto<isize>>(&mut self,b:&mut Tensor<E>,index:I)->Result<&mut Self> where I::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(e,self.get_layout(),"append",None))?;
		self.check_append(b,index)?;
										// wlog the dim of interest to 0 by swapping. swap dims should succeed because we already had to check index validity to check dims
		(self.swap_dims(index,0),b.swap_dims(index,0));
		(self.normalize_layout(),b.normalize_layout());
										// destructure and take ownership of the data to use vec append operation
		let ((mut adata,mut adims),(mut bdata,mut bdims))=(mem::take(self).into_inner(),mem::take(b).into_inner());
		(adata.truncate(adims.dims()[0]*adims.strides()[0].abs() as usize),bdata.truncate(bdims.dims()[0]*bdims.strides()[0].abs() as usize));
										// do the append
		adata.append(&mut bdata);
		adims.dims_mut()[0]+=mem::take(&mut bdims.dims_mut()[0]);
										// return to borrow and undo dim swap
		(*self,*b)=(Self::from_inner(adata,adims),Tensor::from_inner(bdata,bdims));
		( self.swap_dims(index,0),b.swap_dims(index,0));
										// done
		Ok(self)
	}
	#[track_caller]
	/// apply broadcast dim to each index. errors if dims.len()!=self.rank(), or if any of the individual dims fail to broadcast
	pub fn try_broadcast                      (&mut self,dims:impl AsRef<[usize]>)->Result<&mut Self> where E:Clone{
		*self=mem::take(self).into_view_mut().try_broadcast(dims)?.into_tensor();
		Ok(self)
	}
	#[track_caller]
	/// broadcast dim. If the index is out of bounds of the rank, the result is an invalid index error. If dim at the index is 1 and requested size is not 1, the result is a view with the dim equal to the requested size, accomplished by index aliasing the components. If the dim at index is neither 1 nor size, and size is not 1, the result is a mismatch error. If the dim at index is size or size is 1, the result is unchanged from the input
	pub fn try_broadcast_dim<I:TryInto<isize>>(&mut self,axis:I,dim:usize)        ->Result<&mut Self> where E:Clone,I::Error:'static+StdError{
		*self=mem::take(self).into_view_mut().try_broadcast_dim(axis,dim)?.into_tensor();
		Ok(self)
	}
	#[track_caller]
	/// concatenate a collection of tensors along the specified axis
	pub fn try_cat<I:IntoIterator,J:TryInto<isize>>(collection:I,index:J)->Result<Self> where I::Item:Into<Self>,J::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(e,Default::default(),"append",None))?;
		collection.into_iter().map(Into::into).try_fold(None,|mut acc:Option<Self>,mut item|{
			if let Some(a)=acc.as_mut(){
				a.try_append(&mut item,index)?;
			}else{
				acc=Some(item);
			}
			Ok(acc)
		})?.ok_or_else(||Error::empty(Layout::default(),"cat","collection"))
	}
	/// fallible indexing. tries to reference the component, returning an error if the indexing fails. may panic rather than return err if the layout is invalid
	pub fn try_component    <'a,I:Copy+TryInto<isize>>(&'a     self,indices:&[I])->Result<&'a E> where I::Error:'static+StdError{
		let (dims,strides)=(self.dims(),self.strides());
		let (drank,srank)=(dims.len(),strides.len());

		let rank=indices.len();
		(if drank!=rank{return Err(Error::mismatch(drank,self.layout.clone(),"index",rank,None,"rank"))},if srank!=rank{return Err(Error::mismatch(srank,self.layout.clone(),"index",rank,None,"rank"))});

		let off=index::compute_offset(self.dims(),indices,0,self.strides());
		if let Some(c)=off.and_then(|offset|unsafe{std_slice::from_raw_parts(self.ptr,self.len)}.get(offset)){return Ok(c)}

		let mut position=Position::new(indices.len());
		for n in 0..indices.len(){position[n]=indices[n].try_into().map_err(|e|Error::other(e,self.layout.clone(),"index",None))?}

		Err(Error::out_of_bounds(self.layout.clone(),"index",position))
	}
	#[track_caller]
	/// fallible indexing. tries to reference the component, returning an error if the indexing fails. may panic rather than return err if the layout is invalid
	pub fn try_component_mut<'a,I:Copy+TryInto<isize>>(&'a mut self,indices:&[I])->Result<&'a E> where I::Error:'static+StdError{
		let (dims,strides)=(self.dims(),self.strides());
		let (drank,srank)=(dims.len(),strides.len());

		let rank=indices.len();
		(if drank!=rank{return Err(Error::mismatch(drank,self.layout.clone(),"index",rank,None,"rank"))},if srank!=rank{return Err(Error::mismatch(srank,self.layout.clone(),"index",rank,None,"rank"))});

		let off=index::compute_offset(self.dims(),indices,0,self.strides());
		if let Some(c)=off.and_then(|offset|unsafe{std_slice::from_raw_parts_mut(self.ptr,self.len)}.get_mut(offset)){return Ok(c)}

		let mut position=Position::new(indices.len());
		for n in 0..indices.len(){position[n]=indices[n].try_into().map_err(|e|Error::other(e,self.layout.clone(),"index",None))?}

		Err(Error::out_of_bounds(self.layout.clone(),"index",position))
	}
	/// try to flip the axis
	pub fn try_flip_dim     <I:       TryInto<isize>>(    mut self,index:I)     ->Result<Self>  where I::Error:'static+StdError{
		*self.layout_mut()=self.view_ref().try_flip_dim(index)?.get_layout();
		Ok(self)
	}
	#[track_caller]
	/// stack a collection of tensors along a new axis inserted at the index
	pub fn try_stack<I:IntoIterator,J:TryInto<isize>>(collection:I,index:J)->Result<Self> where I::Item:Into<Self>,J::Error:'static+StdError{
		let index=index.try_into().map_err(|e|Error::other(e,Default::default(),"append",None))?;
		collection.into_iter().map(Into::into).try_fold(None,|mut acc:Option<Self>,mut item|{
			item.unsqueeze_dim(index);
			if let Some(a)=acc.as_mut(){
				a.try_append(&mut item,index)?;
			}else{
				acc=Some(item);
			}
			Ok(acc)
		})?.ok_or_else(||Error::empty(Layout::default(),"stack","collection"))
	}
	#[track_caller]
	/// add a dim of size 1
	pub fn unsqueeze_dim(&mut self,dim:impl TryInto<isize>)->&mut Self{
		self.layout=self.view_ref().unsqueeze_dim(dim).get_layout();
		self
	}
	/// get the vc pointer. should not be assumed valid after the tensor is dropped becuase the tensor may drop the view cache too
	pub fn vc(&self)->*const (){
		unsafe{
			let vc=Arc::into_raw(self.viwcnt.clone());
			mem::drop(Arc::from_raw(vc));
			vc
		}
	}
	/// get the current view count
	pub (crate) fn view_count(&self)->usize{Arc::strong_count(&self.viwcnt)}


	/*/// stack the tensors
	pub fn stack<I:IntoIterator>(dim:isize,tensors:I)->Option<Tensor<E>> where I::Item:Into<Tensor<E>>{
		let mut result:Option<Tensor<E>>=None;

		for tensor in tensors{
			let mut tensor=tensor.into();

			if !tensor.unsqueeze_dim(dim){return None}
			if result.is_none(){result=Some(tensor.into())}else if !result.as_mut().unwrap().append(dim,&mut tensor){return None}
		}
		result
	}*/


	/*
	/// fallible concatenation. returns Err with the current state (Option(acc,next),iter) if the operation fails. returns Err with (None,iter) if the iteration produces no values
	pub fn try_cat<I:IntoIterator,R>(dim:isize,tensors:I)->Result<Tensor<E>,(Option<(Tensor<E>,Result<Tensor<E>,R>)>,I::IntoIter)> where I::Item:TryInto<Tensor<E>,Error=R>{// TODO dedicated error enum for try reductions

	}*/
}

#[cfg(feature="serial")]
mod serial{
	impl<'a,E:Deserialize<'a>> Deserialize<'a> for T<E>{
		fn deserialize<D:Deserializer<'a>>(deserializer:D)->StdResult<Self,D::Error>{
			let t=Tensor::deserialize(deserializer)?;
			Ok(T::from_inner(t.data,t.layout))
		}
	}
	impl<E:Serialize> Serialize for T<E>{
		fn serialize<S:Serializer>(&self,serializer:S)->StdResult<S::Ok,S::Error>{
			let ptr=self.as_ptr() as *mut E;
			let len=self.buffer_len();

			unsafe{
				struct ForgetData<E>(Tensor<E>);
				impl<E> Drop for ForgetData<E>{
					fn drop(&mut self){
						mem::forget(mem::take(&mut self.0.data));
					}
				}

				let data=Vec::from_raw_parts(ptr,len,len);
				let layout=self.get_layout();
				let tensor=ForgetData(Tensor{data,layout});

				tensor.0.serialize(serializer)
			}
		}
	}

	#[derive(Deserialize,Serialize)]
	struct Tensor<E>{data:Vec<E>,layout:Layout}

	use serde::{Deserialize,Deserializer,Serialize,Serializer};
	use std::{mem,result::Result as StdResult};
	use super::{Layout,Tensor as T};
}
#[cfg(test)]
mod tests{
	#[test]
	fn broadcast_4x1t4x3(){
		let input:Vec<i32>=vec![1,2,3,4];
		let template:Vec<i32>=vec![1,1,1,2,2,2,3,3,3,4,4,4];

		let mut response:Tensor<i32>=input.clone().into();

		response.unsqueeze_dim(1);
		response.broadcast_dim(1,3);
		assert_eq!(response.into_flat_vec(None),template);

		let response=Tensor::<i32>::from(input).view_ref().unsqueeze_dim(1).broadcast_dim(1,3).into_tensor();
		assert_eq!(response.into_flat_vec(None),template);
	}
	#[test]
	fn indices_1x4t2x2(){
		let tensor:Tensor<char>=Tensor::new(vec!['a','b','c','d'],vec![4]);

		let mut indices=tensor.view_ref().reshape(&[2,2]).indices();

		assert_eq!(indices.len(),4);
		assert_eq!(*indices.next().unwrap(),[0,0]);
		assert_eq!(indices.len(),3);
		assert_eq!(*indices.next().unwrap(),[0,1]);
		assert_eq!(indices.len(),2);
		assert_eq!(*indices.next().unwrap(),[1,0]);
		assert_eq!(indices.len(),1);
		assert_eq!(*indices.next().unwrap(),[1,1]);
		assert_eq!(indices.len(),0);
		assert_eq!(indices.next(),None);
		assert_eq!(indices.len(),0);
		assert_eq!(indices.next(),None);
	}
	#[test]
	fn make_and_view_vector(){
		let data:Vec<i32>=vec![-1,-2,0,1,5];
		let tensor:Tensor<i32>=data.into();

		assert_eq!(tensor.dims(),&[5]);
		assert_eq!(tensor[[0_isize]],-1);
		assert_eq!(tensor[[1_isize]],-2);
		assert_eq!(tensor[[2_isize]],0);
		assert_eq!(tensor[[3_isize]],1);
		assert_eq!(tensor[[4_isize]],5);

		let view=tensor.into_view().slice([1..4].as_slice());
		assert_eq!(view[[0_isize]],-2);
		assert_eq!(view[[1_isize]],0);
		assert_eq!(view[[2_isize]],1);
	}
	#[test]
	fn reshape_1x4t2x2(){
		let tensor:Tensor<char>=Tensor::new(vec!['a','b','c','d'],vec![4]);
		let view=tensor.view_ref().reshape([2,2]);

		assert_eq!(view.dims(),[2,2]);
		assert_eq!(view[[0,0]],'a');
		assert_eq!(view[[1,0]],'c');
		assert_eq!(view[[0,1]],'b');
		assert_eq!(view[[1,1]],'d');

	}
	use super::*;
}

/// tensor value type, similar to multidimensional Vec<T>. tensor uses in place operations by default. convert to a view first with .view() or a similar method to apply operations to views without changing the original structure. layout validity: tensor allows invalid layouts but behaves incorrectly or panics when used for anything, views have guaranteed layout validity
pub struct Tensor<E>{	// TODO serial
	layout:Layout,		// layout info, possibly modified
	viwcnt:Arc<()>,		// count views of the tensor. since not all tensors contained in views have the same lifetime as the reference they're created from, reference counting will be used to know when to free memory
	ptr:*mut E,			// buffer ptr
	len:usize,			// buffer len
	cap:usize,			// buffer capacity
}

use std::{
	borrow::{Borrow,BorrowMut},cmp::{Eq,PartialEq},error::Error as StdError,fmt::{Debug,Formatter,Result as FmtResult},hash::{Hash,Hasher},iter::FromIterator,mem,panic::RefUnwindSafe,ops::{Add,Deref,DerefMut,Index,IndexMut,Range},ptr,slice as std_slice,sync::Arc,sync::atomic::{Ordering::Acquire,self}
};
use super::{Error,GridIter,Layout,Position,Result,View,ViewMut,ViewRef,index};
