impl<'a,E> AsRef<Self   > for ViewRef<'a,E>{
	fn as_ref(&self)->&Self{self}
}
impl<'a,E> AsRef<Self   > for ViewMut<'a,E>{
	fn as_ref(&self)->&Self{self}
}
impl<   E> AsRef<Self   > for View<E>{
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
impl<'a,E:Debug> Debug for ViewRef<'a,E>{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		#[allow(unused)]
		#[derive(Debug)]
		struct ViewRef<'a,E:'a>{data:Vec<&'a E>,dims:&'a [usize]}
		ViewRef{
			data:self.positions().filter_map(|px|self.component(&px).ok()).collect(),
			dims:self.dims()
		}.fmt(f)
	}
}
impl<'a,E:Debug> Debug for ViewMut<'a,E>{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		#[allow(unused)]
		#[derive(Debug)]
		struct ViewMut<'a,E:'a>{data:Vec<&'a E>,dims:&'a [usize]}
		ViewMut{
			data:self.positions().filter_map(|px|self.component(&px).ok()).collect(),
			dims:self.dims()
		}.fmt(f)
	}
}
impl<'a,E:Debug> Debug for View<E>{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		#[allow(unused)]
		#[derive(Debug)]
		struct View<'a,E:'a>{data:Vec<&'a E>,dims:&'a [usize]}
		View{
			data:self.positions().filter_map(|px|self.component(&px).ok()).collect(),
			dims:self.dims()
		}.fmt(f)
	}
}
impl<'a,E> Default  for ViewRef<'a,E>{
	fn default()->Self{empty(1)}
}
impl<'a,E> Default  for ViewMut<'a,E>{
	fn default()->Self{empty_mut(1)}
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
impl<'a,E:'a  > From<Tensor<E>> for ViewRef<'a,E>{
	fn from(inner:Tensor<E>)->Self{
		let (buffer,layout)=inner.into_inner();
		unique(buffer,layout)
	}
}
impl<'a,E:'a  > From<Tensor<E>> for ViewMut<'a,E>{
	fn from(inner:Tensor<E>)->Self{
		let (buffer,layout)=inner.into_inner();
		unique_mut(buffer,layout)
	}
}
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
			self.positions().for_each(|px|self[px].hash(state));
		}else{
			let inner=&self.0[0];
			assert!(inner.buffer_cap()>0);	// TODO this *should* be the case, but I have a feeling we need a slightly stronger unsafe condition somewhere to eliminate this assertion

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
			self.dims()==other.dims()&&self.positions().all(|px|self[&px]==other[&px])
		}else{
			let (inner,other)=(&self.0[0],&other.0[0]);
			assert!(inner.buffer_cap()>0&&other.buffer_cap()>0);	// TODO this *should* be the case, but I have a feeling we need a slightly stronger unsafe condition somewhere to eliminate this assertion

			inner.layout()==other.layout()&&inner.buffer()==other.buffer()
		}
	}
	fn ne(&self,other:&View   <   X>)->bool{
		if self.validate().is_ok(){
			self.dims()!=other.dims()||self.positions().any(|px|self[&px]!=other[&px])
		}else{
			let (inner,other)=(&self.0[0],&other.0[0]);
			assert!(inner.buffer_cap()>0&&other.buffer_cap()>0);	// TODO this *should* be the case, but I have a feeling we need a slightly stronger unsafe condition somewhere to eliminate this assertion

			inner.layout()!=other.layout()||inner.buffer()!=other.buffer()
		}
	}
}
impl<E:Clone> ToOwned for View<E>{
	fn to_owned(&self)->Tens<E>{self.to_tens()}
	type Owned=Tens<E>;
}

impl<'a,E> ViewRef<'a,E>{
	#[track_caller]
	/// broadcast a specific axis. panics if the dims would be incompatible, or if dims and strides have mismatched lengths. panics if the index is out of bounds
	pub fn broadcast_dim(&self,index:impl SignedIndexPosition,rhs:usize)->Self{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().broadcast_dim(index,rhs);
		view
	}
	#[track_caller]
	/// broadcast the dims, panicing if the dims are not broadcast compatible with rhs
	pub fn broadcast<D:AsRef<[usize]>>(&mut self,rhs:D)->Self{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().broadcast(rhs);
		view
	}
	#[track_caller]
	/// reverse the order of components along all axes except the one at the index. panics if the index is out of bounds
	pub fn flip_around(&self,index:impl SignedIndexPosition)->Self{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().flip_around(index);
		view
	}
	#[track_caller]
	/// reverse the order of components along the axis. panics if the index is out of bounds
	pub fn flip_dim(&self,index:impl SignedIndexPosition)->Self{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().flip_dim(index);
		view
	}
	/// reverse the order of components along all axes
	pub fn flip(&self)->Self{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().flip();
		view
	}
	#[track_caller]
	/// slice dim. panics if the index or range are out of bounds
	pub fn slice_dim<I:SignedIndexPosition>(&self,index:impl SignedIndexPosition,range:impl RangeBounds<I>)->Self{
		unsafe{						// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid.
			let mut layout=self.get_layout();
			let mut off=0;
			let mut ptr=self.as_ptr();
			let mut len=self.get_len();

			layout.slice_dim(index,&mut off,range);
			ptr =ptr.add(off);
			len-=off;

			from_raw_parts(layout,ptr,len)
		}
	}
	#[track_caller]
	/// slice. panics if the range are out of bounds
	pub fn slice<I:SignedIndexPosition,R:RangeBounds<I>>(&self,ranges:&[R])->Self{
		unsafe{						// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid.
			let mut layout=self.get_layout();
			let mut off=0;
			let mut ptr=self.as_ptr();
			let mut len=self.get_len();

			layout.slice(&mut off,ranges);
			ptr =ptr.add(off);
			len-=off;

			from_raw_parts(layout,ptr,len)
		}
	}
	#[track_caller]
	/// split the view in two at the position. panics if out of bounds
	pub fn split_at(mut self,index:impl SignedIndexPosition,position:impl SignedIndexPosition)->(Self,Self){
		if self.0.buffer_cap()==0{
			unsafe{					// safety: validity should be maintained. ptr offset will be in bounds since compute_offset panics if out of bounds
				let rank=self.rank();
				let index   =if let Some(ix)=position::unsign_index(index,rank)     {ix}else{panic!("index {} is out of bounds for rank {rank}" ,index   .expect_isize("must be able to convert position to isize"))};

				let dim =self.dims()[index];
				let position=if let Some(px)=position::unsign_position(dim,position){px}else{panic!("position {} is out of bounds for dim {dim}",position.expect_isize("must be able to convert index to isize"))};

				let at=position::compute_offset(&[dim],&[position],&[self.strides()[index]]);
				let rldim=position;
				let mut rllayout=self.get_layout();
				let rlptr=self.as_ptr();
				let rllen=at;
				let rrdim=dim-position;
				let mut rrlayout=self.get_layout();
				let rrptr=rlptr.add(at);
				let rrlen=self.get_len()-at;

				(rllayout.dims_mut()[index],rrlayout.dims_mut()[index])=(rldim,rrdim);
				(from_raw_parts(rllayout,rlptr,rllen),from_raw_parts(rrlayout,rrptr,rrlen))
			}
		}else{
			let rls=self.0.split_off(index,position).into_unique_ref();
			(self,rls)
		}
	}
	#[track_caller]
	/// squeeze an axis of dim 1 into nonexistence. panics if the dim at the index is not equal to 1. panics if out of bounds of the rank
	pub fn squeeze_dim(&self,index:impl SignedIndexPosition)->Self{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().squeeze_dim(index);
		view
	}
	#[track_caller]
	/// swap a pair of axes. unspecified result if the layout is invalid for the buffer
	pub fn swap_dims(&self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->Self{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().swap_dims(a,b);
		view
	}
	/// convert to an owned tensor
	pub fn to_tensor(&self)->Tensor<E> where E:Clone{self.view_ref().into()}
	#[track_caller]
	/// unsqueeze an axis of dim 1 into existence. panics if out of bounds of the rank
	pub fn unsqueeze_dim(&self,index:impl SignedIndexPosition)->Self{
		let mut view=self.clone();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().unsqueeze_dim(index);
		view
	}
	/// reference as a ViewRef
	pub fn view_ref(&self)->ViewRef<'_,E>{
		unsafe{		// preconditions already checked on construction we're just shortening the lifetime
			from_raw_parts(self.get_layout(),self.as_ptr(),self.get_len())
		}
	}
}
impl<'a,E> ViewMut<'a,E>{
	#[track_caller]
	/// broadcast a specific axis. panics if the dims would be incompatible, or if dims and strides have mismatched lengths. panics if the index is out of bounds
	pub fn broadcast_dim(self,index:impl SignedIndexPosition,rhs:usize)->ViewRef<'a,E>{
		let mut view=self.into_view_ref();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().broadcast_dim(index,rhs);
		view
	}
	#[track_caller]
	/// broadcast the dims, panicing if the dims are not broadcast compatible with rhs
	pub fn broadcast<D:AsRef<[usize]>>(self,rhs:D)->ViewRef<'a,E>{
		let mut view=self.into_view_ref();
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().broadcast(rhs);
		view
	}
	#[track_caller]
	/// reverse the order of components along all axes except the one at the index. panics if the index is out of bounds
	pub fn flip_around(self,index:impl SignedIndexPosition)->Self{
		let mut view=self;
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().flip_around(index);
		view
	}
	#[track_caller]
	/// reverse the order of components along the axis. panics if the index is out of bounds
	pub fn flip_dim(self,index:impl SignedIndexPosition)->Self{
		let mut view=self;
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().flip_dim(index);
		view
	}
	/// reverse the order of components along all axes
	pub fn flip(self)->Self{
		let mut view=self;
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().flip();
		view
	}
	/// convert into a shared view
	pub fn into_view_ref(self)->ViewRef<'a,E>{
		unsafe{		// safety: the from raw parts condition for view ref is weaker
			from_raw_parts(self.get_layout(),self.as_ptr(),self.get_len())
		}
	}
	#[track_caller]
	/// slice dim. panics if the index or range are out of bounds
	pub fn slice_dim<I:SignedIndexPosition>(mut self,index:impl SignedIndexPosition,range:impl RangeBounds<I>)->Self{
		unsafe{						// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid.
			let mut layout=self.get_layout();
			let mut off=0;
			let mut ptr=self.as_mut_ptr();
			let mut len=self.get_len();

			layout.slice_dim(index,&mut off,range);
			ptr =ptr.add(off);
			len-=off;

			from_raw_parts_mut(layout,ptr,len)
		}
	}
	#[track_caller]
	/// slice. panics if the range are out of bounds
	pub fn slice<I:SignedIndexPosition,R:RangeBounds<I>>(mut self,ranges:&[R])->Self{
		unsafe{						// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid.
			let mut layout=self.get_layout();
			let mut off=0;
			let mut ptr=self.as_mut_ptr();
			let mut len=self.get_len();

			layout.slice(&mut off,ranges);
			ptr =ptr.add(off);
			len-=off;

			from_raw_parts_mut(layout,ptr,len)
		}
	}
	#[track_caller]
	/// split the view in two at the position. panics if out of bounds
	pub fn split_at(mut self,index:impl SignedIndexPosition,position:impl SignedIndexPosition)->(Self,Self){
		if self.0.buffer_cap()==0{
			unsafe{					// safety: validity should be maintained. ptr offset will be in bounds since compute_offset panics if out of bounds
				let rank=self.rank();
				let index   =if let Some(ix)=position::unsign_index(index,rank)     {ix}else{panic!("index {} is out of bounds for rank {rank}" ,index   .expect_isize("must be able to convert position to isize"))};

				let dim =self.dims()[index];
				let position=if let Some(px)=position::unsign_position(dim,position){px}else{panic!("position {} is out of bounds for dim {dim}",position.expect_isize("must be able to convert index to isize"))};

				let at=position::compute_offset(&[dim],&[position],&[self.strides()[index]]);
				let rldim=position;
				let mut rllayout=self.get_layout();
				let rlptr=self.as_mut_ptr();
				let rllen=at;
				let rrdim=dim-position;
				let mut rrlayout=self.get_layout();
				let rrptr=rlptr.add(at);
				let rrlen=self.get_len()-at;

				(rllayout.dims_mut()[index],rrlayout.dims_mut()[index])=(rldim,rrdim);
				(from_raw_parts_mut(rllayout,rlptr,rllen),from_raw_parts_mut(rrlayout,rrptr,rrlen))
			}
		}else{
			let rls=self.0.split_off(index,position).into_unique_mut();
			(self,rls)
		}
	}
	#[track_caller]
	/// squeeze an axis of dim 1 into nonexistence. panics if the dim at the index is not equal to 1. panics if out of bounds of the rank
	pub fn squeeze_dim(self,index:impl SignedIndexPosition)->Self{
		let mut view=self;
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().squeeze_dim(index);
		view
	}
	#[track_caller]
	/// swap a pair of axes. unspecified result if the layout is invalid for the buffer
	pub fn swap_dims(self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->Self{
		let mut view=self;
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().swap_dims(a,b);
		view
	}
	/// convert to an owned tensor
	pub fn to_tensor(&self)->Tensor<E> where E:Clone{self.view_ref().into()}
	#[track_caller]
	/// unsqueeze an axis of dim 1 into nonexistence. panics if out of bounds of the rank
	pub fn unsqueeze_dim(self,index:impl SignedIndexPosition)->Self{
		let mut view=self;
									// safety: to maintain invariants, ensure layout versions of the ops can't convert valid to invalid
		view.0.layout_mut().squeeze_dim(index);
		view
	}
	/// reference as a ViewRef
	pub fn view_ref(&self)->ViewRef<'_,E>{
		unsafe{		// preconditions already checked on construction we're just shortening the lifetime
			from_raw_parts(self.get_layout(),self.as_ptr(),self.get_len())
		}
	}
	/// reference as a ViewMut
	pub fn view_mut(&mut self)->ViewMut<'_,E>{
		unsafe{		// preconditions already checked on construction we're just shortening the lifetime
			from_raw_parts_mut(self.get_layout(),self.as_mut_ptr(),self.get_len())
		}
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
	#[track_caller]
	/// broadcast a specific axis. panics if the dims would be incompatible, or if dims and strides have mismatched lengths. panics if the index is out of bounds
	pub fn broadcast_dim(&self,index:impl SignedIndexPosition,rhs:usize)->ViewRef<'_,E>{self.view_ref().broadcast_dim(index,rhs)}
	#[track_caller]
	/// broadcasting the dims, panics if the layout is invalid, or if the dims would be incompatible.
	pub fn broadcast<D:AsRef<[usize]>>(&self,rhs:D)->ViewRef<'_,E>{self.view_ref().broadcast(rhs)}
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

		for px in self.positions(){mem.push(self[px].clone())}
		mem
	}
	#[track_caller]
	/// reverse the order of components along all axes except the one at the index. panics if the index is out of bounds
	pub fn flip_around(&self,index:impl SignedIndexPosition)->ViewRef<'_,E>{self.view_ref().flip_dim(index)}
	#[track_caller]
	/// reverse the order of components along the axis. panics if the index is out of bounds
	pub fn flip_dim(&self,index:impl SignedIndexPosition)->ViewRef<'_,E>{self.view_ref().flip_dim(index)}
	/// reverse the order of components along all axes
	pub fn flip(&self)->ViewRef<'_,E>{self.view_ref().flip()}
	/// get the layout
	pub fn get_layout(&self)->Layout{self.0[0].layout().clone()}
	/// get the internal buffer length. Depending on where the View came from, this may be the length a sub-buffer of the actual allocated buffer
	pub fn get_len(&self)->usize{self.0[0].buffer_len()}
	/// checks if the dims contain 0
	pub fn is_empty(&self)->bool{self.dims().contains(&0)}
	/// check if the layout is normalized. also checks that the count is equal to the buffer len
	pub fn is_layout_normalized(&self)->bool{
		let layout=self.get_layout();
		layout.is_normalized()&&layout.count()==self.get_len()
	}
	/// check if this layout represents a scalar.
	pub fn is_scalar(&self)->bool{self.dims().is_empty()}
	/// count the buffer length. assumes layout validity and may overflow if invalid
	pub fn len(&self)->usize{self.0[0].layout().len()}
	#[track_caller]
	/// apply a function to every component, returning a new tensor. panics if the layout is invalid for the buffer
	pub fn map<F:FnMut(&E)->Y,Y>(&self,mut f:F)->Tens<Y>{
		let buffer=if self.is_layout_normalized(){
			unsafe{		// safety: if a normalized layout with count==len meets this type's invariants, ptr must be valid for references up to ptr+len
				let ptr=self.as_ptr();
				let len=self.get_len();
						// with the layout normalized, this has predictable iteration order
				slice::from_raw_parts(ptr,len).iter().map(f).collect()
			}
		}else{
			error::unwrap_or_panic(self.validate());

			let mut b=Vec::with_capacity(self.count());
			for px in self.positions(){b.push(f(&self[px]))}

			b
		};				// the result will have normalized layout
		let layout=Layout::new(self.dims());

		Tens::from_inner(buffer,layout)
	}
	/// iterate over the positions in the tensor
	pub fn positions(&self)->PositionIter{PositionIter::new(self.dims())}
	/// return the tensor rank. unspecified result if dims and strides have different ranks
	pub fn rank(&self)->usize{self.0[0].layout().rank()}
	#[track_caller]
	/// slice dim. panics if the index or range are out of bounds or if the layout is invalid for the buffer
	pub fn slice_dim<I:SignedIndexPosition>(&self,index:impl SignedIndexPosition,range:impl RangeBounds<I>)->ViewRef<'_,E>{self.view_ref().slice_dim(index,range)}
	#[track_caller]
	/// slice. panics if the range are out of bounds or if the layout is invalid for the buffer
	pub fn slice<I:SignedIndexPosition,R:RangeBounds<I>>(&self,ranges:&[R])->ViewRef<'_,E>{self.view_ref().slice(ranges)}
	#[track_caller]
	/// split the view in two at the position. panics if out of bounds or invalid layout
	pub fn split_at(&self,index:impl SignedIndexPosition,position:impl SignedIndexPosition)->(ViewRef<'_,E>,ViewRef<'_,E>){self.view_ref().split_at(index,position)}
	#[track_caller]
	/// split the view in two at the position. panics if out of bounds or invalid mutable layout
	pub fn split_at_mut(&mut self,index:impl SignedIndexPosition,position:impl SignedIndexPosition)->(ViewMut<'_,E>,ViewMut<'_,E>){self.view_mut().split_at(index,position)}
	#[track_caller]
	/// squeeze an axis of dim 1 into nonexistence. panics if the dim at the index is not equal to 1. panics if out of bounds of the rank or if the layout is invalid
	pub fn squeeze_dim(&self,index:impl SignedIndexPosition)->ViewRef<'_,E>{self.view_ref().squeeze_dim(index)}
	/// reference the dims
	pub fn strides(&self)->&[isize]{self.0[0].layout().strides()}
	#[track_caller]
	/// swap a pair of axes. unspecified result if the layout is invalid for the buffer
	pub fn swap_dims(&self,a:impl SignedIndexPosition,b:impl SignedIndexPosition)->ViewRef<'_,E>{self.view_ref().swap_dims(a,b)}
	/// convert to an owned tensor
	pub fn to_tensor(&self)->Tensor<E> where E:Clone{self.to_tens().tensor()}
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
	#[track_caller]
	/// unsqueeze an axis of dim 1 into existence. panics if out of bounds of the rank or if the layout is invalid
	pub fn unsqueeze_dim(&self,index:impl SignedIndexPosition)->ViewRef<'_,E>{self.view_ref().unsqueeze_dim(index)}
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

/// create an empty view of a particular rank. panics if rank is 0
pub fn empty<'a,E:'a>(rank:usize)->ViewRef<'a,E>{
	let (buffer,layout)=Tens::empty(rank).into_inner();
	unique(buffer,layout)
}
/// create an empty view of a particular rank. panics if rank is 0
pub fn empty_mut<'a,E:'a>(rank:usize)->ViewMut<'a,E>{
	let (buffer,layout)=Tens::empty(rank).into_inner();
	unique_mut(buffer,layout)
}
/// create a view reference from raw parts. The layout must be valid for len, and any offset of ptr reachable through a position in bounds of the layout must be simultaneously valid as a shared reference for lifetime 'a.
pub unsafe fn from_raw_parts<'a,E:'a>(layout:Layout,ptr:*const E,len:usize)->ViewRef<'a,E>{
	unsafe{		// safety: ViewRef meets the borrowed buffer contition. cap==0, so we don't need ptr and len to form a vec. The outer layout validity condition implies the inner one.
		ViewRef(Tens::_from_raw_parts(layout,ptr as *mut E,len,0),PhantomData)
	}
}
/// create a view reference from raw parts. The layout must be valid for len, and any offset of ptr reachable through a position in bounds of the layout must be simultaneously valid as a mutable reference for lifetime 'a.
pub unsafe fn from_raw_parts_mut<'a,E:'a>(layout:Layout,ptr:*mut E,len:usize)->ViewMut<'a,E>{
	unsafe{		// safety: ViewRef meets the borrowed buffer contition. cap==0, so we don't need ptr and len to form a vec. The outer layout validity condition implies the inner one.
		ViewMut(Tens::_from_raw_parts(layout,ptr,len,0),PhantomData)
	}
}
/// create a view ref of a scalar reference
pub fn scalar<E>(data:&E)->ViewRef<'_,E>{
	unsafe{		// safety: vector layout is valid for vector, reference has appropriate lifetime
		from_raw_parts(Layout::scalar(),data,1)
	}
}
/// create a view mut of a scalar reference
pub fn scalar_mut<E>(data:&mut E)->ViewMut<'_,E>{
	unsafe{		// safety: vector layout is valid for vector, reference has appropriate lifetime
		from_raw_parts_mut(Layout::scalar(),data,1)
	}
}
#[track_caller]
/// create a view reference with owned data. panics if the layout is not valid for the data
pub fn unique<'a,E:'a>(mut buffer:Vec<E>,layout:Layout)->ViewRef<'a,E>{
	error::unwrap_or_panic(layout.validate(buffer.len()).map_err(|e|e.with_op("unique")));
	unsafe{		// safety: layout validity has been checked, pointer validity is ensured by Vec
		let ptr=buffer.as_mut_ptr();
		let len=buffer.len();
		let cap=buffer.capacity();

		ViewRef(Tens::from_raw_parts(layout,ptr,len,cap),PhantomData)
	}
}
#[track_caller]
/// create a view reference with owned data. panics if the layout is not mutably valid for the data
pub fn unique_mut<'a,E:'a>(mut buffer:Vec<E>,layout:Layout)->ViewMut<'a,E>{
	error::unwrap_or_panic(layout.validate_mut(buffer.len()).map_err(|e|e.with_op("unique")));
	unsafe{		// safety: layout validity has been checked, pointer validity is ensured by Vec
		let ptr=buffer.as_mut_ptr();
		let len=buffer.len();
		let cap=buffer.capacity();

		ViewMut(Tens::from_raw_parts(layout,ptr,len,cap),PhantomData)
	}
}
/// create a view ref of a vector reference
pub fn vector<E>(data:&[E])->ViewRef<'_,E>{
	unsafe{		// safety: vector layout is valid for vector, reference has appropriate lifetime
		let len=data.len();
		from_raw_parts(Layout::new([len]),data.as_ptr(),len)
	}
}
/// create a view ref of a vector reference
pub fn vector_mut<E>(data:&mut [E])->ViewMut<'_,E>{
	unsafe{		// safety: vector layout is valid for vector, reference has appropriate lifetime
		let len=data.len();
		from_raw_parts_mut(Layout::new([len]),data.as_mut_ptr(),len)
	}
}

#[repr(transparent)]
/// a shared tensor view. Shared layout validity is guaranteed as an invariant
pub struct ViewRef<'a,E:'a>(Tens<E>,PhantomData<&'a E>);
#[repr(transparent)]
/// a mutable tensor view. Mutable layout validity is guaranteed as an invariant
pub struct ViewMut<'a,E:'a>(Tens<E>,PhantomData<&'a E>);
#[repr(transparent)]
/// A dynamically-sized view into a multidimensional sequence. Layout validity is lazily checked, and producing a View with a layout invalid for its buffer is allowed, but failing to maintain shared validity may lead to panics or unexpected behavior of some functions.
pub struct View<E>([Tens<E>]);// Do not make Sized. Although it looks like this could be Sized as all usages of it currently have the same dynamic size, implementing Sized for this would be unsound. See mem::swap. It would be nicer to make this less fake by making it a slice of MaybeUninit<E> with layout fields at the beginning, but after experimenting with that I've found it has questionable soundess implications for mutable view splitting. Maybe I'll think of a better way next time I refactor this library

use std::{
	borrow::{Borrow,BorrowMut,ToOwned},cmp::{Eq,PartialEq},fmt::{Debug,Formatter,Result as FmtResult},hash::{Hash,Hasher},marker::PhantomData,ops::{Deref,DerefMut,Index,IndexMut,RangeBounds},slice
};
use super::{
	Error,Layout,PositionIter,Position,Result,Tens,error,position::{SignedIndexPosition,self},tensor::Tensor
};
