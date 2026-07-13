impl<'a,E> AsRef<Self> for ViewRef<'a,E>{
	fn as_ref(&self)->&Self{self}
}
impl<'a,E> AsRef<Self> for ViewMut<'a,E>{
	fn as_ref(&self)->&Self{self}
}
impl<   E> AsRef<Self> for View<E>{
	fn as_ref(&self)->&Self{self}
}
impl<'a,E> AsRef<View<E>> for ViewRef<'a,E>{
	fn as_ref(&    self)->&    View<E>{self.0.as_view()}
}
impl<'a,E> AsRef<View<E>> for ViewMut<'a,E>{
	fn as_ref(&    self)->&    View<E>{self.0.as_view()}
}
impl<'a,E> AsMut<Self   > for ViewRef<'a,E>{
	fn as_mut(&mut self)->&mut Self{self}
}
impl<'a,E> AsMut<Self   > for ViewMut<'a,E>{
	fn as_mut(&mut self)->&mut Self{self}
}
impl<   E> AsMut<Self   > for View<E>{
	fn as_mut(&mut self)->&mut Self{self}
}
impl<'a,E> AsMut    <View<E>> for ViewMut<'a,E>{
	fn as_mut(&mut     self)->&mut View<E>{self.0.as_mut_view()}
}
impl<'a,E> Borrow   <View<E>> for ViewRef<'a,E>{
	fn borrow(&        self)->&    View<E>{self.0.as_view()}
}
impl<'a,E> Borrow   <View<E>> for ViewMut<'a,E>{
	fn borrow(&        self)->&    View<E>{self.0.as_view()}
}
impl<'a,E> BorrowMut<View<E>> for ViewMut<'a,E>{
	fn borrow_mut(&mut self)->&mut View<E>{self.0.as_mut_view()}
}
impl<'a,E> Clone for ViewRef<'a,E>{
	fn clone(&self)->Self{
		unsafe{		// safety: viewref is constructed with the same preconditions
			from_raw_parts(self.0.get_layout(),self.0.as_ptr(),self.0.buffer_len())
		}
	}
}
impl<'a,E> Default  for ViewRef<'a,E>{
	fn default()->Self{Self::empty(1)}
}
impl<'a,E> Default  for ViewMut<'a,E>{
	fn default()->Self{Self::empty(1)}
}
impl<'a,E> Deref    for ViewRef<'a,E>{
	fn deref(&self)->&Self::Target{self.0.deref()}
	type Target=View<E>;
}
impl<'a,E> Deref    for ViewMut<'a,E>{
	fn deref(&self)->&Self::Target{self.0.deref()}
	type Target=View<E>;
}
impl<'a,E> DerefMut for ViewMut<'a,E>{
	fn deref_mut(&mut self)->&mut Self::Target{self.0.deref_mut()}
}
impl<'a,E:Eq  > Eq   for ViewRef<'a,E>{}
impl<'a,E:Eq  > Eq   for ViewMut<'a,E>{}
impl<   E:Eq  > Eq   for View   <   E>{}
impl<'a,E:Hash> Hash for ViewRef<'a,E>{
	fn hash<H:Hasher>(&self,state:&mut H){(**self).hash(state)}
}
impl<'a,E:Hash> Hash for ViewMut<'a,E>{
	fn hash<H:Hasher>(&self,state:&mut H){(**self).hash(state)}
}
impl<   E:Hash> Hash for View   <   E>{
	fn hash<H:Hasher>(&self,state:&mut H){
		if self.validate().is_ok(){
			self.dims().hash(state);
			self.positions().for_each(|ix|self[ix].hash(state));
		}else{
			let inner=&self.0[0];
			assert!(inner.buffer_cap()>0);	// TODO I have a feeling we need a slightly stronger unsafe condition somewhere to eliminate this assertion

			inner.layout().hash(state);
			inner.buffer().hash(state);
		}
	}
}
impl<'a,E,P:SignedIndexPosition> Index<&[P]> for ViewRef<'a,E>{
	fn index(&self,position:&[P])->&Self::Output{self.as_view().index(position)}
	type Output=E;
}
impl<'a,E,P:SignedIndexPosition> Index<&[P]> for ViewMut<'a,E>{
	fn index(&self,position:&[P])->&Self::Output{self.as_view().index(position)}
	type Output=E;
}
impl<   E,P:SignedIndexPosition> Index<&[P]> for View<E>{
	#[track_caller]
	fn index(&self,position:&[P])->&Self::Output{
		let (dims,strides)=(self.dims(),self.strides());
		// error::unwrap_or_panic(error::check_bounds(dims,position).map_err(|e|e.with_op("index"))); // since we're panicing on out of bounds anyway, just use compute_offset's panic

		unsafe{		// safety: "valid layouts (in general not just for buffer) stored by the resulting Tens must never produce an offset less than len for which ptr+offset not valid for conversion to a shared reference" - Tens precondition. "If cap>0, (ptr, len, cap) must be ok to convert to Vec." - Tens precondition. View with cap==0 are created in from_raw_parts, from_raw_parts_mut, or empty, which require valid layouts, and any offset of ptr reachable through a position in bounds of the layout must be simultaneously valid as a shared reference. View with cap>0 are created from Tens with cap>0 condition, which implies anywhere in the buffer can be shared referenced.
			let offset=position::compute_offset(dims,position,strides);
			let ptr=self.as_ptr();

			if offset>self.get_len(){return error::unwrap_or_panic(Err(Error::invalid_layout(self.get_layout(),"index")))}
			&    *ptr.add(offset)
		}
	}
	type Output=E;
}
impl<'a,E,P:SignedIndexPosition,const N:usize> Index<[P;N]> for ViewRef<'a,E>{
	#[track_caller]
	fn index(&self,index:[P;N])->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<'a,E,P:SignedIndexPosition,const N:usize> Index<[P;N]> for ViewMut<'a,E>{
	#[track_caller]
	fn index(&self,index:[P;N])->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<   E,P:SignedIndexPosition,const N:usize> Index<[P;N]> for View<E>{
	#[track_caller]
	fn index(&self,index:[P;N])->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<'a,E> Index< Position> for ViewRef<'a,E>{
	#[track_caller]
	fn index(&self,index:Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<'a,E> Index< Position> for ViewMut<'a,E>{
	#[track_caller]
	fn index(&self,index:Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<   E> Index< Position> for View<E>{
	#[track_caller]
	fn index(&self,index:Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<'a,E> Index<&Position> for ViewRef<'a,E>{
	#[track_caller]
	fn index(&self,index:&Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<'a,E> Index<&Position> for ViewMut<'a,E>{
	#[track_caller]
	fn index(&self,index:&Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<   E> Index<&Position> for View<E>{
	#[track_caller]
	fn index(&self,index:&Position)->&Self::Output{self.index(index.as_slice())}
	type Output=E;
}
impl<'a,E,P:SignedIndexPosition> IndexMut<&[P]> for ViewMut<'a,E>{
	#[track_caller]
	fn index_mut(&mut self,position:&[P])->&mut Self::Output{self.as_mut_view().index_mut(position)}
}
impl<   E,P:SignedIndexPosition> IndexMut<&[P]> for View<E>{
	#[track_caller]
	fn index_mut(&mut self,position:&[P])->&mut Self::Output{
		let (dims,strides)=(self.dims(),self.strides());
		// error::unwrap_or_panic(error::check_bounds(dims,position).map_err(|e|e.with_op("index"))); // since we're panicing on out of bounds anyway, just use compute_offset's panic

		unsafe{		// safety: "valid layouts (in general not just for buffer) stored by the resulting Tens must never produce an offset less than len for which ptr+offset not valid for conversion to a shared reference, and if the buffer will be mutated, temporary conversion to a mutable reference" - Tens precondition. "If cap>0, (ptr, len, cap) must be ok to convert to Vec." - Tens precondition. View with cap==0 are created in from_raw_parts, from_raw_parts_mut, or empty. from_raw_parts_mut requires valid layouts, and any offset of ptr reachable through a position in bounds of the layout must be simultaneously valid as a mutable reference, and from_raw_parts returns a ViewRef which only allows shared View references. View with cap>0 are created from Tens with cap>0 condition, which implies anywhere in the buffer can be mutable referenced.
			let offset=position::compute_offset(dims,position,strides);
			let ptr=self.as_mut_ptr();

			if offset>self.get_len(){return error::unwrap_or_panic(Err(Error::invalid_layout(self.get_layout(),"index")))}
			&mut *ptr.add(offset)
		}
	}
}
impl<'a,E,P:SignedIndexPosition,const N:usize> IndexMut<[P;N]> for ViewMut<'a,E>{
	#[track_caller]
	fn index_mut(&mut self,index:[P;N])->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<   E,P:SignedIndexPosition,const N:usize> IndexMut<[P;N]> for View<E>{
	#[track_caller]
	fn index_mut(&mut self,index:[P;N])->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<'a,E> IndexMut< Position> for ViewMut<'a,E>{
	#[track_caller]
	fn index_mut(&mut self,index:Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<   E> IndexMut< Position> for View<E>{
	#[track_caller]
	fn index_mut(&mut self,index:Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<'a,E> IndexMut<&Position> for ViewMut<'a,E>{
	#[track_caller]
	fn index_mut(&mut self,index:&Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<   E> IndexMut<&Position> for View<E>{
	#[track_caller]
	fn index_mut(&mut self,index:&Position)->&mut Self::Output{self.index_mut(index.as_slice())}
}
impl<'a   ,E:PartialEq<X>,X> PartialEq<Tensor <X    >> for ViewRef<'a,E>{
	fn eq(&self,other:&Tensor <   X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&Tensor <   X>)->bool{(**self)!=(**other)}
}
impl<'a   ,E:PartialEq<X>,X> PartialEq<Tensor <X    >> for ViewMut<'a,E>{
	fn eq(&self,other:&Tensor <   X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&Tensor <   X>)->bool{(**self)!=(**other)}
}
impl<      E:PartialEq<X>,X> PartialEq<Tensor <X    >> for View<E>{
	fn eq(&self,other:&Tensor <   X>)->bool{(*self)==(**other)}
	fn ne(&self,other:&Tensor <   X>)->bool{(*self)!=(**other)}
}
impl<'a   ,E:PartialEq<X>,X> PartialEq<Tens   <X    >> for ViewRef<'a,E>{
	fn eq(&self,other:&Tens   <   X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&Tens   <   X>)->bool{(**self)!=(**other)}
}
impl<'a   ,E:PartialEq<X>,X> PartialEq<Tens   <X   >> for ViewMut<'a,E>{
	fn eq(&self,other:&Tens   <   X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&Tens   <   X>)->bool{(**self)!=(**other)}
}
impl<      E:PartialEq<X>,X> PartialEq<Tens   <X   >> for View<E>{
	fn eq(&self,other:&Tens   <   X>)->bool{( *self)==(**other)}
	fn ne(&self,other:&Tens   <   X>)->bool{( *self)!=(**other)}
}
impl<'a,'b,E:PartialEq<X>,X> PartialEq<ViewRef<'b,X>> for ViewRef<'a,E>{
	fn eq(&self,other:&ViewRef<'b,X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&ViewRef<'b,X>)->bool{(**self)!=(**other)}
}
impl<'a,'b,E:PartialEq<X>,X> PartialEq<ViewRef<'b,X>> for ViewMut<'a,E>{
	fn eq(&self,other:&ViewRef<'b,X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&ViewRef<'b,X>)->bool{(**self)!=(**other)}
}
impl<   'b,E:PartialEq<X>,X> PartialEq<ViewRef<'b,X>> for View<E>{
	fn eq(&self,other:&ViewRef<'b,X>)->bool{( *self)==(**other)}
	fn ne(&self,other:&ViewRef<'b,X>)->bool{( *self)!=(**other)}
}
impl<'a,'b,E:PartialEq<X>,X> PartialEq<ViewMut<'b,X>> for ViewRef<'a,E>{
	fn eq(&self,other:&ViewMut<'b,X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&ViewMut<'b,X>)->bool{(**self)!=(**other)}
}
impl<'a,'b,E:PartialEq<X>,X> PartialEq<ViewMut<'b,X>> for ViewMut<'a,E>{
	fn eq(&self,other:&ViewMut<'b,X>)->bool{(**self)==(**other)}
	fn ne(&self,other:&ViewMut<'b,X>)->bool{(**self)!=(**other)}
}
impl<   'b,E:PartialEq<X>,X> PartialEq<ViewMut<'b,X>> for View<E>{
	fn eq(&self,other:&ViewMut<'b,X>)->bool{( *self)==(**other)}
	fn ne(&self,other:&ViewMut<'b,X>)->bool{( *self)!=(**other)}
}
impl<'a   ,E:PartialEq<X>,X> PartialEq<View   <   X>> for ViewRef<'a,E>{
	fn eq(&self,other:&View   <   X>)->bool{(**self)==( *other)}
	fn ne(&self,other:&View   <   X>)->bool{(**self)!=( *other)}
}
impl<'a   ,E:PartialEq<X>,X> PartialEq<View   <   X>> for ViewMut<'a,E>{
	fn eq(&self,other:&View   <   X>)->bool{(**self)==( *other)}
	fn ne(&self,other:&View   <   X>)->bool{(**self)!=( *other)}
}
impl<      E:PartialEq<X>,X> PartialEq<View  <    X>> for View<E>{
	fn eq(&self,other:&View   <   X>)->bool{
		if self.validate().is_ok(){
			self.dims()==other.dims()&&self.positions().all(|ix|self[&ix]==other[&ix])
		}else{
			let (inner,other)=(&self.0[0],&other.0[0]);
			assert!(inner.buffer_cap()>0&&other.buffer_cap()>0);	// TODO I have a feeling we need a slightly stronger unsafe condition somewhere to eliminate this assertion

			inner.layout()==other.layout()&&inner.buffer()==other.buffer()
		}
	}
	fn ne(&self,other:&View   <   X>)->bool{
		if self.validate().is_ok(){
			self.dims()!=other.dims()||self.positions().any(|ix|self[&ix]!=other[&ix])
		}else{
			let (inner,other)=(&self.0[0],&other.0[0]);
			assert!(inner.buffer_cap()>0&&other.buffer_cap()>0);	// TODO I have a feeling we need a slightly stronger unsafe condition somewhere to eliminate this assertion

			inner.layout()!=other.layout()||inner.buffer()!=other.buffer()
		}
	}
}
impl<E:Clone> ToOwned for View<E>{
	fn to_owned(&self)->Tens<E>{self.to_tens()}
	type Owned=Tens<E>;
}

impl<'a,E> ViewRef<'a,E>{
	/// create an empty view of a particular rank. panics if rank is 0
	pub fn empty(rank:usize)->Self{
		assert!(rank!=0);
		unsafe{		// safety: trivial
			let (buffer,layout)=Tens::empty(rank).into_inner();
			let ptr=buffer.as_ptr();
			let len=0;

			from_raw_parts(layout,ptr,len)
		}
	}
	#[track_caller]
	/// swap a pair of axes. unspecified result if the layout is invalid for the buffer
	pub fn swap_dims(&self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->ViewRef<'a,E>{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().swap_dims(a,b);
		view
	}
}
impl<'a,E> ViewMut<'a,E>{
	/// create an empty view of a particular rank. panics if rank is 0
	pub fn empty(rank:usize)->Self{
		assert!(rank!=0);
		unsafe{		// safety: trivial
			let (mut buffer,layout)=Tens::empty(rank).into_inner();
			let ptr=buffer.as_mut_ptr();
			let len=0;

			from_raw_parts_mut(layout,ptr,len)
		}
	}
	#[track_caller]
	/// swap a pair of axes. unspecified result if the layout is invalid for the buffer
	pub fn swap_dims(self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->ViewMut<'a,E>{
		let mut view=self;
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().swap_dims(a,b);
		view
	}
}
impl<E> View<E>{
	/// get the pointer to the buffer. Depending on where the View came from, this may be the pointer to a sub-buffer of the actual allocated buffer
	pub fn as_mut_ptr(&mut self)->*mut E{self.0[0].as_mut_ptr()}
	/// get the pointer to the buffer. Depending on where the View came from, this may be the pointer to a sub-buffer of the actual allocated buffer
	pub fn as_ptr(&self)->*const E{self.0[0].as_ptr()}
	/// reference as a View
	pub fn as_view(&self)->&View<E>{self}
	/// reference as a View
	pub fn as_mut_view(&mut self)->&mut View<E>{self}
	/// reference a component of the tensor. This checks bounds, but does not explicitly check layout validity. It will however return an invalid layout error if the position passes the bounds check but still fails to produce an offset in bounds of the buffer
	pub fn component<P:SignedIndexPosition>(&self,position:&[P])->Result<&E>{
		let (dims,strides)=(self.dims(),self.strides());
		error::check_bounds(dims,position).map_err(|e|e.with_op("index"))?;

		unsafe{		// safety: "valid layouts (in general not just for buffer) stored by the resulting Tens must never produce an offset less than len for which ptr+offset not valid for conversion to a shared reference" - Tens precondition. "If cap>0, (ptr, len, cap) must be ok to convert to Vec." - Tens precondition. View with cap==0 are created in from_raw_parts, from_raw_parts_mut, or empty, which require valid layouts, and any offset of ptr reachable through a position in bounds of the layout must be simultaneously valid as a shared reference. View with cap>0 are created from Tens with cap>0 condition, which implies anywhere in the buffer can be shared referenced.
			let offset=position::compute_offset(dims,position,strides);
			let ptr=self.as_ptr();

			if offset>self.get_len(){return Err(Error::invalid_layout(self.get_layout(),"index"))}
			Ok(&    *ptr.add(offset))
		}
	}
	/// reference a component of the tensor. This checks bounds, but does not explicitly check layout validity. It will however return an invalid layout error if the position passes the bounds check but still fails to produce an offset in bounds of the buffer
	pub fn component_mut<P:SignedIndexPosition>(&mut self,position:&[P])->Result<&mut E>{
		let (dims,strides)=(self.dims(),self.strides());
		error::check_bounds(dims,position).map_err(|e|e.with_op("index"))?;

		unsafe{		// safety: "valid layouts (in general not just for buffer) stored by the resulting Tens must never produce an offset less than len for which ptr+offset not valid for conversion to a shared reference, and if the buffer will be mutated, temporary conversion to a mutable reference" - Tens precondition. "If cap>0, (ptr, len, cap) must be ok to convert to Vec." - Tens precondition. View with cap==0 are created in from_raw_parts, from_raw_parts_mut, or empty. from_raw_parts_mut requires valid layouts, and any offset of ptr reachable through a position in bounds of the layout must be simultaneously valid as a mutable reference, and from_raw_parts returns a ViewRef which only allows shared View references. View with cap>0 are created from Tens with cap>0 condition, which implies anywhere in the buffer can be mutable referenced.
			let offset=position::compute_offset(dims,position,strides);
			let ptr=self.as_mut_ptr();

			if offset>self.get_len(){return Err(Error::invalid_layout(self.get_layout(),"index"))}
			Ok(&mut *ptr.add(offset))
		}
	}
	/// count the number of components. assumes layout validity and may overflow if invalid
	pub fn count(&self)->usize{self.0[0].layout().count()}
	/// reference the dims
	pub fn dims(&self)->&[usize]{self.0[0].layout().dims()}
	/// flatten into a vec. panics if the layout is invalid for the buffer
	pub fn flat_vec(&self,mem:impl Into<Option<Vec<E>>>)->Vec<E> where E:Clone{
		let mut mem=mem.into().unwrap_or_default();
		mem.reserve(self.count());

		for ix in self.positions(){mem.push(self[ix].clone())}
		mem
	}
	/// get the layout
	pub fn get_layout(&self)->Layout{self.0[0].layout().clone()}
	/// get the internal buffer length. Depending on where the View came from, this may be the length a sub-buffer of the actual allocated buffer
	pub fn get_len(&self)->usize{self.0[0].buffer_len()}
	/// count the buffer length. assumes layout validity and may overflow if invalid
	pub fn len(&self)->usize{self.0[0].layout().len()}
	/// iterate over the positions in the tensor
	pub fn positions(&self)->PositionIter{PositionIter::new(self.dims())}
	/// reference the dims
	pub fn strides(&self)->&[isize]{self.0[0].layout().strides()}
	#[track_caller]
	/// swap a pair of axes. unspecified result if the layout is invalid for the buffer
	pub fn swap_dims(&self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->ViewRef<'_,E>{self.view_ref().swap_dims(a,b)}
	/// convert to an owned tensor
	pub fn to_tens(&self)->Tens  <E> where E:Clone{
		if self.0[0].buffer_cap()>0{
			unsafe{		// safety: "If cap>0, (ptr, len, cap) must be ok to convert to Vec" - precondition on Tens construction. If this is the case, (ptr, len) should be fine as a slice
				let ptr=self.as_ptr();
				let len=self.get_len();
												// if this is an owned tensor, reproduce its layout exactly
				let data=slice::from_raw_parts(ptr,len);
				let layout=self.0[0].layout().clone();
				let buffer=data.to_vec();

				Tens::from_inner(buffer,layout)
			}
		}else{
			let buffer=self.flat_vec(None);
			let layout=Layout::new(self.dims());
												// if this is a view of a borrowed tensor, we can expect validity but can't reproduce the exact buffer configuration. This should be fine since we consider valid tensors equal if they have the same components at the same positions
			Tens::from_inner(buffer,layout)
		}
	}
	/// reference as a ViewRef. Err if the layout is invalid for the buffer
	pub fn try_view_mut(&mut self)->Result<ViewMut<'_,E>>{
		unsafe{		// safety: ptr validity is ensured by construction of self. layout validity is ensured by the check
			let (layout,len)=(self.get_layout(),self.get_len());
			let ptr=self.as_mut_ptr();

			layout.validate_mut(len).map_err(|e|e.with_op("view"))?;
			Ok(from_raw_parts_mut(layout,ptr,len))
		}
	}
	/// reference as a ViewRef. Err if the layout is invalid for the buffer
	pub fn try_view_ref(&self)->Result<ViewRef<'_,E>>{
		unsafe{		// safety: ptr validity is ensured by construction of self. layout validity is ensured by the check
			let (layout,len)=(self.get_layout(),self.get_len());
			let ptr=self.as_ptr();

			layout.validate(len).map_err(|e|e.with_op("view"))?;
			Ok(from_raw_parts(layout,ptr,len))
		}
	}
	/// check validity of the layout for the buffer len
	pub fn validate(&self)->Result<()>{self.get_layout().validate(self.get_len())}
	/// check validity of the layout for the buffer len
	pub fn validate_mut(&self)->Result<()>{self.get_layout().validate_mut(self.get_len())}
	#[track_caller]
	/// reference as a ViewRef. panic if the layout is invalid for the buffer
	pub fn view_ref(&self)->ViewRef<'_,E>{error::unwrap_or_panic(self.try_view_ref())}
	#[track_caller]
	/// reference as a ViewRef. panic if the layout is invalid for the buffer
	pub fn view_mut(&mut self)->ViewMut<'_,E>{error::unwrap_or_panic(self.try_view_mut())}
	/// reference as a View
	pub fn view(&self)->&View<E>{self.as_view()}
}

/// create a view reference from raw parts. The layout must be valid for len, and any offset of ptr reachable through a position in bounds of the layout must be simultaneously valid as a shared reference for lifetime 'a.
pub unsafe fn from_raw_parts<'a,E:'a>(layout:Layout,ptr:*const E,len:usize)->ViewRef<'a,E>{
	unsafe{		// safety: ViewRef meets the borrowed buffer contition. cap==0, so we don't need ptr and len to form a vec. The outer layout validity condition implies the inner one.
		ViewRef(Tens::_from_raw_parts(layout,ptr as *mut E,len,0),PhantomData)
	}
}/// create a view reference from raw parts. The layout must be valid for len, and any offset of ptr reachable through a position in bounds of the layout must be simultaneously valid as a mutable reference for lifetime 'a.
pub unsafe fn from_raw_parts_mut<'a,E:'a>(layout:Layout,ptr:*mut E,len:usize)->ViewMut<'a,E>{
	unsafe{		// safety: ViewRef meets the borrowed buffer contition. cap==0, so we don't need ptr and len to form a vec. The outer layout validity condition implies the inner one.
		ViewMut(Tens::_from_raw_parts(layout,ptr,len,0),PhantomData)
	}
}

#[repr(transparent)]
/// a shared tensor view
pub struct ViewRef<'a,E:'a>(Tens<E>,PhantomData<&'a E>);
#[repr(transparent)]
/// a mutable tensor view
pub struct ViewMut<'a,E:'a>(Tens<E>,PhantomData<&'a E>);
#[repr(transparent)]
/// A dynamically-sized view into a multidimensional sequence
pub struct View<E>([Tens<E>]);// Do not make Sized. Although it looks like this could be Sized as all usages of it currently have the same dynamic size, implementing Sized for this would be unsound. See mem::swap. It would be nicer to make this less fake by making it a slice of MaybeUninit<E> with layout fields at the beginning, but after experimenting with that I've found it has questionable soundess implications for mutable view splitting. Maybe I'll think of a better way next time I refactor this library

use std::{
	borrow::{Borrow,BorrowMut,ToOwned},cmp::{Eq,PartialEq},hash::{Hash,Hasher},marker::PhantomData,ops::{Deref,DerefMut,Index,IndexMut,Range},slice
};
use super::{
	Error,Layout,PositionIter,Position,Result,Tens,error,position::{SignedIndexPosition,self},tensor::Tensor
};
