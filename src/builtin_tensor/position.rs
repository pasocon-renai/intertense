impl AsMut<[isize]> for Position{
	fn as_mut(&mut self)->&mut [isize]{Arc::make_mut(&mut self.0)}
}
impl AsMut<Position> for Position{
	fn as_mut(&mut self)->&mut Position{self}
}
impl AsRef<[isize]> for Position{
	fn as_ref(&self)->&[isize]{self.0.as_ref()}
}
impl AsRef<Position> for Position{
	fn as_ref(&self)->&Position{self}
}
impl Borrow<[isize]> for Position{
	fn borrow(&self)->&[isize]{&*self}
}
impl BorrowMut<[isize]> for Position{
	fn borrow_mut(&mut self)->&mut [isize]{&mut *self}
}
impl Clone for Position{
	fn clone(&self)->Self{Self(self.0.clone())}
	fn clone_from(&mut self,other:&Self){
		if self.rank()==other.rank(){self.copy_from_slice(other)}
		else{self.0.clone_from(&other.0)}
	}
}
impl Deref for Position{
	fn deref(&self)->&Self::Target{self.as_ref()}
	type Target=[isize];
}
impl DerefMut for Position{
	fn deref_mut(&mut self)->&mut Self::Target{self.as_mut()}
}
impl Display for Position{
	fn fmt(&self,f:&mut Formatter<'_>)->FmtResult{
		write!(f,"[")?;
		for x in self.iter().take(self.len().wrapping_sub(1)){
			write!(f,"{x}, ")?;
		}
		for x in self.last().into_iter(){
			write!(f,"{x}")?;
		}
		']'.fmt(f)
	}
}
impl DoubleEndedIterator for PositionIntoIter{
	fn rfold<B,F:FnMut(B,Self::Item)->B>(self,init:B,mut f:F)->B{self.state.rfold(init,|acc,ix|f(acc,self.position[ix]))}
	fn next_back(&mut self)->Option<Self::Item>{self.state.next_back().map(|ix|self.position[ix])}
	fn nth_back(&mut self,n:usize)->Option<Self::Item>{self.state.nth_back(n).map(|ix|self.position[ix])}
}
impl DoubleEndedIterator for PositionIter{
	fn next_back(&mut self)->Option<Self::Item>{
		let dims=self.layout.dims();
		let rank=dims.len();

		if let Some(back)=self.back.as_deref_mut(){
			if decrement_position(dims,back)>0{
				increment_position(dims,back);
				return None
			}
		}else if !dims.contains(&0){
			self.back=Some(Position::end(rank));
		}
		if let Some(front)=self.front.as_deref()&&let Some(back)=self.back.as_deref_mut(){
			if greater_equals_position(dims,front,back){
				increment_position(dims,back);
				return None
			}
		}
		self.back.clone()
	}
	fn nth_back(&mut self,n:usize)->Option<Self::Item>{
		let dims=self.layout.dims();
		let rank=dims.len();
												// if n is 0 return next, the len calculation that takes about as long as next's worst case
		if n==0{return self.next_back()}
		let max=if let Some(m)=self.len().checked_sub(1){m}else{return None};
												// halt if n>=self.len()-1. This simplifies check for done and also implicitly handles the edge case where n==usize::MAX, which would otherwise overflow n+1 later. Empty shape case is handled previously
		if n>=max{
			let item=if n==max{self.back.clone()}else{None};
			self.back.clone_from(&self.front);

			if let Some(back)=self.back.as_deref_mut(){
				decrement_position(dims,back);
			}
			return item
		}										// the bounds are exclusive. creating a bound counts as a next
		if let Some(back)=self.back.as_deref_mut(){
			rewind_position(dims,n+1,back);
		}else{
			let mut back=Position::new(rank);
			rewind_position(dims,n,&mut back);

			self.back=Some(back)
		}
		self.back.clone()
	}
}
impl ExactSizeIterator for PositionIntoIter{
	fn len(&self)->usize{self.state.len()}
}
impl ExactSizeIterator for PositionIter{
	fn len(&self)->usize{
		let mut acc=0;
		let mut borrow=0;
		let dims=self.dims();
		let mut placevalue=1;
		let rank=dims.len();
												// to handle the exclusiveness and optionalness of the bounds, adjust iterators for None case by adding an extra index at the front. positions that are present will have a 1 at this index. an absent front position will have a 0 at this index, so it can indicate the last item on the previous line. an absent back position will have a 2 at this index, so it can indicate the first item on the next line
		let dims=dims.iter().copied().rev().chain([3]);
		let front=self.front.as_deref().into_iter().flat_map(|px|{
			px.iter().copied().rev()   .chain([1])
		}).chain(
			iter::repeat(-1).take(rank).chain([0])
		);
		let back=self.back.as_deref().into_iter().flat_map(|px|{
			px.iter().copied().rev()   .chain([1])
		}).chain(
			iter::repeat( 0).take(rank).chain([2])
		);
												// do something like mixed radix subtraction accumulate the length from the difference between the bounds
		for ((dim,front),back) in dims.zip(front).zip(back){
			if dim==0{return 0}

			let front=unsign_position(dim,front).unwrap()+borrow;
			let back =unsign_position(dim,back ).unwrap();

			borrow=if front>back{1}else{0};

			acc+=(borrow*dim+back-front)*placevalue;
			placevalue*=dim;
		}										// since the bounds are exclusive, the accumulated difference is off by 1
		if borrow>0{0}else{acc.saturating_sub(1)}
	}
}
impl FromIterator<isize> for Position{
	fn from_iter<I:IntoIterator<Item=isize>>(iter:I)->Self{
		let v:Vec<isize>=iter.into_iter().collect();
		Self(Arc::from(v))
	}
}
impl<const N:usize> From<[isize;N]> for Position{
	fn from(value:[isize;N])->Self{
		Self(Arc::from(value.as_slice()))
	}
}
impl From<&[isize]> for Position{
	fn from(value:&[isize])->Self{
		Self(Arc::from(value))
	}
}
impl<I:SignedIndexPosition> Index<I> for Position{
	#[track_caller]
	fn index(&self,index:I)->&Self::Output{
		let ix=index.expect_isize("index must be convertable to isize");
		let rank=self.len();

		let index=unsign_index(index,rank);
		if let Some(ix)=index{
			&self.0[ix]
		}else{
			panic!("index {ix} is out of bounds for rank {rank}")
		}
	}
	type Output=isize;
}
impl<I:SignedIndexPosition> IndexMut<I> for Position{
	#[track_caller]
	fn index_mut(&mut self,index:I)->&mut Self::Output{
		let ix=index.expect_isize("index must be convertable to isize");
		let rank=self.len();

		let index=unsign_index(index,rank);
		if let Some(ix)=index{
			&mut Arc::make_mut(&mut self.0)[ix]
		}else{
			panic!("index {ix} is out of bounds for rank {rank}")
		}
	}
}
impl<'a> IntoIterator for &'a Position{
	fn into_iter(self)->Self::IntoIter{self.iter()}
	type IntoIter=SliceIter<'a,isize>;
	type Item=&'a isize;
}
impl<'a> IntoIterator for &'a mut Position{
	fn into_iter(self)->Self::IntoIter{self.iter_mut()}
	type IntoIter=SliceIterMut<'a,isize>;
	type Item=&'a mut isize;
}
impl IntoIterator for Position{
	fn into_iter(self)->Self::IntoIter{
		let position=self;
		let state=0..position.len();

		PositionIntoIter{position,state}
	}
	type IntoIter=PositionIntoIter;
	type Item=isize;
}
impl Iterator for PositionIntoIter{
	fn fold<B,F:FnMut(B,Self::Item)->B>(self,init:B,mut f:F)->B{self.state.fold(init,|acc,ix|f(acc,self.position[ix]))}
	fn next(&mut self)->Option<Self::Item>{self.state.next().map(|ix|self.position[ix])}
	fn nth(&mut self,n:usize)->Option<Self::Item>{self.state.nth(n).map(|ix|self.position[ix])}
	fn size_hint(&self)->(usize,Option<usize>){self.state.size_hint()}
	type Item=isize;
}
impl Iterator for PositionIter{
	fn next(&mut self)->Option<Self::Item>{
		let dims=self.layout.dims();
		let rank=dims.len();

		if let Some(front)=self.front.as_deref_mut(){
			if increment_position(dims,front)>0{
				decrement_position(dims,front);
				return None;
			}
		}else if !dims.contains(&0){
			self.front=Some(Position::new(rank));
		}
		if let Some(front)=self.front.as_deref_mut()&&let Some(back)=self.back.as_deref(){
			if greater_equals_position(dims,front,back){
				decrement_position(dims,front);
				return None
			}
		}
		self.front.clone()
	}
	fn nth(&mut self,n:usize)->Option<Self::Item>{
		let dims=self.layout.dims();
		let rank=dims.len();
												// if n is 0 return next, the len calculation that takes about as long as next's worst case
		if n==0{return self.next()}
		let max=if let Some(m)=self.len().checked_sub(1){m}else{return None};
												// halt if n>=self.len()-1. This simplifies check for done and also implicitly handles the edge case where n==usize::MAX, which would otherwise overflow n+1 later. Empty shape case is handled previously
		if n>=max{
			let item=if n==max{self.front.clone()}else{None};
			self.front.clone_from(&self.back);

			if let Some(front)=self.front.as_deref_mut(){
				decrement_position(dims,front);
			}
			return item
		}										// the bounds are exclusive. creating a bound counts as a next
		if let Some(front)=self.front.as_deref_mut(){
			advance_position(dims,n+1,front);
		}else{
			let mut front=Position::new(rank);
			advance_position(dims,n,&mut front);

			self.front=Some(front)
		}
		self.front.clone()
	}
	fn size_hint(&self)->(usize,Option<usize>){
		let len=self.len();
		(len,Some(len))
	}
	type Item=Position;
}
impl<P:SignedIndexPosition> PartialEq<(&[P],&[usize])> for Position{
	fn eq(&self,other:&(&[P],&[usize]))->bool{self.partial_cmp(other)==Some(Ordering::Equal)}
	fn ne(&self,other:&(&[P],&[usize]))->bool{self.partial_cmp(other)!=Some(Ordering::Equal)}
}
impl<P:SignedIndexPosition> PartialEq<(&[P],&Layout)> for Position{
	fn eq(&self,other:&(&[P],&Layout))->bool{*self==(other.0,&**other.1.dims())}
	fn ne(&self,other:&(&[P],&Layout))->bool{*self!=(other.0,&**other.1.dims())}
}
impl PartialEq for Position{
	fn eq(&self,other:&Self)->bool{self.partial_cmp(other)==Some(Ordering::Equal)}
	fn ne(&self,other:&Self)->bool{self.partial_cmp(other)!=Some(Ordering::Equal)}
}
impl<P:SignedIndexPosition> PartialOrd<(&[P],&[usize])> for Position{
	fn partial_cmp(&self,other:&(&[P],&[usize]))->Option<Ordering>{
		let (qx,dims)=other;
		let px=self.as_slice();

		let rank=dims.len();
		if px.len()!=rank||qx.len()!=rank{return None}

		for ix in 0..rank{
			let dim=dims[ix];
			let px=if let Some(px)=unsign_position(dim,px[ix]){px}else{return None};
			let qx=if let Some(qx)=unsign_position(dim,qx[ix]){qx}else{return None};

			match px.cmp(&qx){
				Ordering::Equal=>continue,
				o=>return Some(o)
			}
		}
		Some(Ordering::Equal)
	}
}
impl<P:SignedIndexPosition> PartialOrd<(&[P],&Layout)> for Position{
	fn partial_cmp(&self,other:&(&[P],&Layout))->Option<Ordering>{self.partial_cmp(&(other.0,&**other.1.dims()))}
}
impl PartialOrd for Position{
	fn partial_cmp(&self,other:&Self)->Option<Ordering>{
		let (px,qx)=(self.as_slice(),other.as_slice());
		let rank=px.len();

		if qx.len()!=rank{return None}

		for ix in 0..rank{
			match px[ix].cmp(&qx[ix]){
				Ordering::Equal=>continue,
				o=>return Some(o)
			}
		}
		Some(Ordering::Equal)
	}
}
impl<I:Copy+TryFrom<isize>+TryInto<isize>> SignedIndexPosition for I{}
impl TryFrom<Layout> for PositionIter{
	fn try_from(layout:Layout)->Result<Self>{
		if let Err(e)=error::check_dims(layout.dims()){return Err(e.with_op("positions"))}

		Ok(Self{
			layout,
			front:None,back:None
		})
	}
	type Error=Error;
}
impl TryFrom<&[usize]> for PositionIter{
	fn try_from(dims:&[usize])->Result<Self>{
		if let Err(e)=error::check_dims(dims){return Err(e.with_op("positions"))}

		Ok(Self{
			layout:Layout::from_inner(dims.to_vec(),Vec::new()),
			front:None,back:None
		})
	}
	type Error=Error;
}

impl PositionIter{
	/// returns current back position, which is the most recent position from next_back or None if the iteration has ended and returned None from either end. If next_back has not been called, this will either return None or another sentinel value
	pub fn back(&self)->Option<Position>{self.back.clone()}
	/// references the dimensions of the tensor whose position this iterates over
	pub fn dims(&self)->&[usize]{self.layout.dims()}
	/// returns current front position, which is the most recent position from next or None if the iteration has ended and returned None from either end. If next has not been called, this may return None or another sentinel value
	pub fn front(&self)->Option<Position>{self.front.clone()}
	#[track_caller]
	/// creates a new position iter over the specified dimensions. The iterator will iterate once over every position in bounds of dims, with positive positions from forward iteration and negative positions from reverse iteration. panics if the dims are invalid (if any exceed isize::MAX, or their product overflows a usize)
	pub fn new<D:AsRef<[usize]>>(dims:D)->Self{
		match Self::try_from(dims.as_ref()){
			Err(e)=>panic!("{e}"),
			Ok(i)=>i
		}
	}
	#[track_caller]
	/// create iterator with the dimensions and starting point. from is included. to isn't. Note that the returned iterator will contain all positions between from and to in a full position iteration, rather than only positions geometrically between from and to. Coordinate signs given in the input will be conserved with from signs mapping to forward iteration and to signs mapping to reverse iteration, however, the signs corresponding to None inputs are unspecified. panics if the dims are invalid (if any exceed isize::MAX, or their product overflows a usize)
	pub fn range(dims:impl AsRef<[usize]>,from:impl Into<Option<Position>>,to:impl Into<Option<Position>>)->Self{// TODO ensure back is not less than front
		let dims=dims.as_ref();
		if let Err(e)=error::check_dims(dims){panic!("{}",e.with_op("positions"))}
														// convert options
		let (mut start,mut stop)=(from.into(),to.into());
														// check bounds
		if let Some(s)=start.as_ref()&&let Err(e)=error::check_bounds(dims,s){panic!("{}",e.with_op("positions"))}
		if let Some(s)=stop .as_ref()&&let Err(e)=error::check_bounds(dims,s){panic!("{}",e.with_op("positions"))}
														// make empty if bounds are equal
		if let Some(s)=start.as_ref()&&let Some(z)=stop.as_ref()&&equals_position(dims,s,z){
			(start,stop)=(None,None);
		}												// decrement start since this is internally double exclusive
		if let Some(s)=start.as_mut(){
			if decrement_position(dims,s)>0{start=None}
		}

		let layout=Layout::from_inner(dims.to_vec(),Vec::new());
		Self{layout,front:start,back:stop}
	}
	/// references the rank of the tensor whose position this iterates over
	pub fn rank(&self)->usize{self.layout.rank()}
}
impl Position{
	/// advance position
	pub fn advance(&mut self,dims:&[usize],distance:usize)->usize{advance_position(dims,distance,self)}
	/// references as a slice
	pub fn as_slice(&self)->&[isize]{self.0.as_ref()}
	/// references as an unsigned coordinates. note that this doesn't perform index normalization, it just casts isize to usize
	pub fn cast_unsigned(&self)->&[usize]{
		let ix:&[isize]=self.as_ref();
		unsafe{slice::from_raw_parts(ix.as_ptr() as *const usize,ix.len())}
	}
	/// advance position by -1
	pub fn decrement(&mut self,dims:&[usize])->usize{decrement_position(dims,self)}
	/// creates a new position whose coordinates are all -1
	pub fn end(rank:usize)->Self{Self(vec![-1;rank].into())}
	/// advance position by 1
	pub fn increment(&mut self,dims:&[usize])->usize{increment_position(dims,self)}
	/// convert the indices to their unsigned forms
	pub fn into_unsigned(mut self,dims:&[usize])->Self{
		self.unsign(dims);
		self
	}
	/// creates a new zeroed position
	pub fn new(rank:usize)->Self{Self(vec![0;rank].into())}
	/// return the rank. more vocab consistent than using self.len() through deref
	pub fn rank(&self)->usize{self.len()}
	/// rewind position
	pub fn rewind(&mut self,dims:&[usize],distance:usize)->usize{rewind_position(dims,distance,self)}
	/// create a position from coordinates. returns none if any of the coordinates can't be converted to isize
	pub fn try_from_coordinates<P:SignedIndexPosition>(coordinates:&[P])->Option<Self>{
		let mut position=Vec::with_capacity(coordinates.len());
		for &px in coordinates{
			position.push(px.try_into().ok()?);
		}

		Some(Self(Arc::from(position.as_slice())))
	}
	/// create a position from an iter of optional isize.  returns none if any of the coordinates can't be unwrapped
	pub fn try_from_iter(coordinates:impl IntoIterator<Item=Option<isize>>)->Option<Self>{
		let coordinates=coordinates.into_iter();

		let mut position=Vec::with_capacity(coordinates.size_hint().0);
		for px in coordinates{
			position.push(px?);
		}

		Some(Self(Arc::from(position.as_slice())))
	}
	/// convert the indices to their unsigned forms
	pub fn unsign(&mut self,dims:&[usize])->Option<&[usize]>{
		assert_eq!(dims.len(),self.len());

		for (&dim,index) in dims.iter().zip(self.iter_mut()){*index=unsign_position(dim,*index)? as isize}
		Some(self.cast_unsigned())
	}
}

#[cfg(test)]
mod tests{
	#[test]
	fn position_offset(){
		let dims=vec![4,6];
		let strides=vec![2,10];

		assert_eq!(compute_offset(&dims,&[2,3],&strides),34);
	}
	#[test]
	fn position_iter_collect(){
		let expected:Vec<[isize;2]>=vec![[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]];
		let response:Vec<[isize;2]>=PositionIter::new([2,3]).map(|px|[px[0],px[1]]).collect();

		assert_eq!(expected,response);
	}
	#[test]
	fn position_iter_len(){
		let expected:Vec<usize>=vec![6,5,4,3,2,1,0];
		let mut iter=PositionIter::new([2,3]).map(|px|[px[0],px[1]]);
		let response:Vec<usize>=(0..7).map(|_|{
			let l=iter.len();
			if rand::random(){iter.next()}else{iter.next_back()};

			l
		}).collect();

		assert_eq!(expected,response);
		assert_eq!(iter.next(),None);

		let expected:Vec<usize>=vec![12,10,8,6,4,2,0];
		let mut iter=PositionIter::new([4,3]).map(|px|[px[0],px[1]]);
		let response:Vec<usize>=(0..7).map(|_|{
			let l=iter.len();
			if rand::random(){iter.nth(1)}else{iter.nth_back(1)};

			l
		}).collect();

		assert_eq!(expected,response);
		assert_eq!(iter.next(),None);
	}
	#[test]
	fn position_iter_scalar(){
		let mut iter=PositionIter::new([]);

		assert_eq!(iter.len(),1);
		assert_eq!(iter.next(),Some(Position::new(0)));
		assert_eq!(iter.next(),None);
		assert_eq!(iter.next(),None);
	}

	use super::*;
}

#[track_caller]
/// advance by distance along the last axis. When the end of an axis is reached, loops around and advances the position along the next left axis. Panics if any coordinates fail to convert to isize or fail to convert back to their original type after updating. Returns a 'carry' value of how much a hypothetical next-left axis beyond what this function can see would have to be advanced to complete the operation. Bounds and dims are not explicitly checked, but it is expected that no dim exceeds isize::MAX, and their product does not overflow a usize, and that the position is in bounds. If those invariants are upheld, assuming numeric conversions for the position type are well-behaved, the signs of position will be preserved. Since count and by extension distance may be any usize, advance and rewind are separated rather than taking a signed distance. This function relies on division so consider using increment/decrement position for distances of +-1.
pub fn advance_position<P:SignedIndexPosition>(dims:&[usize],mut distance:usize,position:&mut [P])->usize{
													// this algorithm is similar to add assignment of a usize to a big endian mixed radix number
	for (&dim,coordinate) in dims.iter().rev().zip(position.iter_mut().rev()){
		if distance==0{break}
		let (carry,updatedcoordinate);
													// get adjustment to the current coordinate
		let dx=distance%dim;
		let px=coordinate.expect_isize("coordinates must fit in isize");
													// split by sign to avoid overflow. Due to isize::MIN having greater magnitude than isize::MAX, match signedness to the case rather than taking any absolute value
		if px<0{
			let dx=dx as isize;
			let qx=dx+px;

			carry=qx>0;
			updatedcoordinate=qx-if carry{dim}else{0} as isize;
		}else{
			let px=px as usize;
			let qx=dx+px;

			carry=qx>dim;
			updatedcoordinate=(qx-if carry{dim}else{0}) as isize;
		}
													// adjust position and calculate the distance for the next axis
		*coordinate=updatedcoordinate.expect_coordinate("updated coordinates must fit in their original type");
		distance=carry as usize+distance/dim;
	}
	distance
}
/// computes the length of a buffer required to hold a tensor with the given dims and strides. This expects an equal number of dims and strides, and that no dim exceeds isize::MAX, and that the buffer len doesn't exceed isize::MAX, but it doesn't check. Use error::checked_len to check the dims for validity while computing the length. If any dim is 0, the result is 0. Otherwise, each axis independently contributes (dim-1)*|stride| to the maximum offset. The required buffer length is 1 more than the maximum offset
pub fn buffer_len(dims:&[usize],strides:&[isize])->usize{
	if dims.contains(&0){return 0}
	dims.iter().rev().zip(strides.iter().rev()).fold(1,|acc,(&dim,&stride)|acc+(dim-1)*stride.abs() as usize)
}
/// casts an unsigned position to a signed position without changing its bits
pub fn cast_to_signed(position:&[usize])->&[isize]{
	let rank=position.len();
	unsafe{slice::from_raw_parts(position.as_ptr() as *const isize,rank)}
}
/// casts an unsigned position to a signed position without changing its bits
pub fn cast_to_signed_mut(position:&mut [usize])->&mut [isize]{
	let rank=position.len();
	unsafe{slice::from_raw_parts_mut(position.as_ptr() as *mut isize,rank)}
}
/// casts a signed position to an unsigned position without normalizing it
pub fn cast_to_unsigned(position:&[isize])->&[usize]{
	let rank=position.len();
	unsafe{slice::from_raw_parts(position.as_ptr() as *const usize,rank)}
}
/// casts a signed position to an unsigned position without normalizing it
pub fn cast_to_unsigned_mut(position:&mut [isize])->&mut [usize]{
	let rank=position.len();
	unsafe{slice::from_raw_parts_mut(position.as_ptr() as *mut usize,rank)}
}
#[track_caller]
/// compare if two positions refer to the same component. panics if out of bounds or rank mismatch
pub fn compare_position<P:SignedIndexPosition,Q:SignedIndexPosition>(dims:&[usize],px:&[P],qx:&[Q])->Ordering{
	let rank=dims.len();
	if px.len()!=rank||qx.len()!=rank{panic!("mismatched rank")}

	for ix in 0..rank{
		let dim=dims[ix];
		let px=if let Some(px)=unsign_position(dim,px[ix]){px}else{panic!("out of bounds")};
		let qx=if let Some(qx)=unsign_position(dim,qx[ix]){qx}else{panic!("out of bounds")};

		match px.cmp(&qx){
			Ordering::Equal=>continue,
			o=>return o
		}
	}
	Ordering::Equal
}
/// counts the components by taking the product of the dims
pub fn component_count(dims:&[usize])->usize{dims.iter().product()}
#[track_caller]
/// computes the offset of a component. panics if out of bounds
pub fn compute_offset<I:SignedIndexPosition>(dims:&[usize],position:&[I],strides:&[isize])->usize{
	dims.iter().rev().zip(position.iter().rev()).zip(strides.iter().rev()).fold(0,|acc,((&dim,px),&stride)|{
		let mut px=px.expect_isize("coordinates must fit in isize");
		assert!(px>=-(dim as isize)&&px<dim as isize);

		if stride<0{px=!px}
		if px<0{px+=dim as isize}

		acc+px as usize*stride.abs() as usize
	})
}
/// rewind the position by 1. see rewind_position for more details
pub fn decrement_position<P:SignedIndexPosition>(dims:&[usize],position:&mut [P])->usize{
	for (&dim,coordinate) in dims.iter().rev().zip(position.iter_mut().rev()){
		let px=(*coordinate).try_into().ok().expect("coordinates must fit in isize");
		let range=if px<0{-(dim as isize)..0}else{0..dim as isize};

		let carry=px==range.start;
		let qx=if carry{range.end}else{px}-1;

		*coordinate=qx.try_into().ok().expect("updated coordinates must fit in their original type");
		if !carry{return 0}
	}
	1
}
#[track_caller]
/// compare if two positions refer to the same component. panics if out of bounds or rank mismatch
pub fn equals_position<P:SignedIndexPosition,Q:SignedIndexPosition>(dims:&[usize],px:&[P],qx:&[Q])->bool{compare_position(dims,px,qx)==Ordering::Equal}
/// internal iteration over positions that lacks the overhead of the tricks PositionIter and Position use to avoid cloning. the iteration will start if the dims don't contain 0. the iteration will stop after reaching a state where each position is one less than the corresponding dim, when it rolls over to all 0. dims should be <= isize::MAX
pub fn for_positions<F:FnMut(&mut [P])->ControlFlow<()>,P:SignedIndexPosition>(dims:&[usize],mut f:F,position:&mut [P]){
	assert_eq!(dims.len(),position.len());
	if dims.contains(&0){return}

	if f(position)==ControlFlow::Break(()){return}
	loop{
		if increment_position(dims,position)>0{break}
		if f(position)==ControlFlow::Break(()){break}
	}
}
#[track_caller]
/// compare if two positions refer to the same component. panics if out of bounds or rank mismatch
pub fn greater_equals_position<P:SignedIndexPosition,Q:SignedIndexPosition>(dims:&[usize],px:&[P],qx:&[Q])->bool{compare_position(dims,px,qx)!=Ordering::Less}
#[track_caller]
/// compare if two positions refer to the same component. panics if out of bounds or rank mismatch
pub fn greater_position<P:SignedIndexPosition,Q:SignedIndexPosition>(dims:&[usize],px:&[P],qx:&[Q])->bool{compare_position(dims,px,qx)==Ordering::Greater}
/// advance the position by 1. see advance_position for more details
pub fn increment_position<P:SignedIndexPosition>(dims:&[usize],position:&mut [P])->usize{
	for (&dim,coordinate) in dims.iter().rev().zip(position.iter_mut().rev()){
		let px=(*coordinate).try_into().ok().expect("coordinates must fit in isize");
		let range=if px<0{-(dim as isize)..0}else{0..dim as isize};

		let carry=px==range.end-1;
		let qx=if carry{range.start}else{px+1};

		*coordinate=qx.try_into().ok().expect("updated coordinates must fit in their original type");
		if !carry{return 0}
	}
	1
}
#[track_caller]
/// compare if two positions refer to the same component.
pub fn less_equals_position<P:SignedIndexPosition,Q:SignedIndexPosition>(dims:&[usize],px:&[P],qx:&[Q])->bool{compare_position(dims,px,qx)!=Ordering::Greater}
#[track_caller]
/// compare if two positions refer to the same component.
pub fn less_position<P:SignedIndexPosition,Q:SignedIndexPosition>(dims:&[usize],px:&[P],qx:&[Q])->bool{compare_position(dims,px,qx)==Ordering::Less}
#[track_caller]
/// rewind by distance along the last axis. When the beginning of an axis is reached, loops around and rewinds the position along the next left axis. Panics if any coordinates fail to convert to isize or fail to convert back to their original type after updating. Returns a 'carry' value of how much a hypothetical next-left axis beyond what this function can see would have to be rewound to complete the operation. Bounds and dims are not explicitly checked, but it is expected that no dim exceeds isize::MAX, and their product does not overflow a usize, and that the position is in bounds. If those invariants are upheld, assuming numeric conversions for the position type are well-behaved, the signs of position will be preserved. Since count and by extension distance may be any usize, advance and rewind are separated rather than taking a signed distance. This function relies on division so consider using increment/decrement position for distances of +-1.
pub fn rewind_position<P:SignedIndexPosition>(dims:&[usize],mut distance:usize,position:&mut [P])->usize{
													// this algorithm is similar to sub assignment of a usize to a big endian mixed radix number
	for (&dim,coordinate) in dims.iter().rev().zip(position.iter_mut().rev()){
		if distance==0{break}
		let (carry,updatedcoordinate);
													// get adjustment to the current coordinate
		let dx=distance%dim;
		let px=(*coordinate).try_into().ok().expect("coordinates must fit in isize");
													// split by sign to avoid overflow, matching the signedness to the case similarly to advance position. However, since this is a rewind, the signedness will be opposite
		if px>=0{
			let dx=dx as isize;
			let qx=px-dx;

			carry=qx<0;
			updatedcoordinate=qx+if carry{dim}else{0} as isize;
		}else{
			let px=(-px) as usize;
			let qx=px+dx;

			carry=qx>dim;
			updatedcoordinate=-((qx-if carry{dim}else{0}) as isize);
		}
													// adjust position and calculate the distance for the next axis
		*coordinate=updatedcoordinate.try_into().ok().expect("updated coordinates must fit in their original type");
		distance=carry as usize+distance/dim;
	}
	distance
}
/// computes an unsigned position component given a signed position component. If the position is < -dim, >= dim, or not castable to isize, returns None. If position<0, returns the value of dim+position. Otherwise, returns the value of position
pub fn unsign_position(dim:usize,position:impl SignedIndexPosition)->Option<usize>{
									// cast
	let position=position.try_into().ok()?;
									// special cases
	if position< -(dim as isize){return None}
	if position>=  dim as isize {return None}
									// normalize
	if position<0{return Some((dim as isize+position) as usize)}
	Some(position as usize)
}
/// computes the unsigned index given a signed index. If the index is < -rank, >= rank, or not castable to isize, returns None. If index<0, returns the value of dim+index. Otherwise, returns the value of index
pub fn unsign_index(index:impl SignedIndexPosition,rank:usize)->Option<usize>{unsign_position(rank,index)}
/// computes the unsigned start or end of range given a signed end of range. unlike unsign_position(dim,stop), when dim==stop, this returns dim rather than none
pub fn unsign_range_bound(dim:usize,rb:impl SignedIndexPosition)->Option<usize>{
	let rb=rb.try_into().ok()?;
	if dim as isize==rb{return Some(dim)}

	unsign_position(dim,rb)
}
#[track_caller]
/// unsign a slice of positions
pub fn unsign_position_slice<'a,P:SignedIndexPosition>(dims:&[usize],positions:&'a mut [P])->Option<&'a mut [P]>{
	for (&dim,px) in dims.iter().zip(positions.iter_mut()){*px=P::try_from(unsign_position(dim,*px)? as isize).ok()?}
	Some(positions)
}

#[derive(Clone,Debug,Default)]
/// iterates over positions in a tensor
pub struct PositionIter{layout:Layout,front:Option<Position>,back:Option<Position>}
#[derive(Debug,Default)]
#[cfg_attr(feature="serial",derive(Deserialize,Serialize))]
#[repr(transparent)]
/// wraps a signed tensor position stored as a reference counted slice of isize. Note that since the positions don't contain the dims, two positions alone cannot always be determined to refer or not refer to the same component in the case of negative positioning. Therefore, despite their PartialOrd implementation, one may want to compare positions using the position module's comparison functions instead, which include a dims argument
pub struct Position(Arc<[isize]>);
#[derive(Clone,Debug,Default)]
/// into iterator for Position
pub struct PositionIntoIter{position:Position,state:Range<usize>}

/// bounds required for a type to be a signed index or position. These types should cleanly convert to and from isize, but TryFrom/TryInto are used instead of From/Into for convenience when using other integer types. Occasionally this means error results when converting the index may be possible, which may lead to out of bounds errors unexpectedly containing empty positions
pub trait SignedIndexPosition:Copy+TryFrom<isize>+TryInto<isize>{
	#[track_caller]
	/// shortcut for self.expect_isize().try_into().ok().expect()
	fn expect_coordinate<T:SignedIndexPosition>(self,details:&str)->T{self.expect_isize(details).try_into().ok().expect(details)}
	#[track_caller]
	/// shortcut for self.try_into().ok().expect()
	fn expect_isize(self,details:&str)->isize{self.try_into().ok().expect(details)}
}

#[cfg(feature="serial")]
use serde::{Deserialize,Serialize};
use std::{
	borrow::{Borrow,BorrowMut},cmp::{Ord,Ordering,PartialEq,PartialOrd},fmt::{Display,Formatter,Result as FmtResult},iter::{FromIterator,self},ops::{ControlFlow,Deref,DerefMut,Index,IndexMut,Range},slice::{Iter as SliceIter,IterMut as SliceIterMut,self},sync::Arc
};
use super::{Error,Layout,Result,error};
