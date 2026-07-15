impl AsMut<[isize]> for Position{
	fn as_mut(&mut self)->&mut [isize]{self.as_mut_slice()}
}
impl AsMut<Position> for Position{
	fn as_mut(&mut self)->&mut Position{self}
}
impl AsRef<[isize]> for Position{
	fn as_ref(&self)->&[isize]{self.as_slice()}
}
impl AsRef<Position> for Position{
	fn as_ref(&self)->&Position{self}
}
impl Borrow<[isize]> for Position{
	fn borrow(&self)->&[isize]{self.as_slice()}
}
impl BorrowMut<[isize]> for Position{
	fn borrow_mut(&mut self)->&mut [isize]{self.as_mut_slice()}
}
impl Clone for Position{
	fn clone(&self)->Self{Self(self.0.clone())}
	fn clone_from(&mut self,other:&Self){
		if self.rank()==other.rank(){self.copy_from_slice(other)}
		else{self.0.clone_from(&other.0)}
	}
}
impl Deref for Position{
	fn deref(&self)->&Self::Target{self.as_slice()}
	type Target=[isize];
}
impl DerefMut for Position{
	fn deref_mut(&mut self)->&mut Self::Target{self.as_mut_slice()}
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
										// if the iteration has started, decrement the back position. Otherwise, start from the last position. (unless dims contains 0 because empty tensors produce no positions)
		if let Some(back)=self.back.as_deref_mut(){
			if decrement_position(dims,back)>0{
										// if it wraps the iteration is done and it's reached the end. undo the decrement to make it not restart after the None.
				increment_position(dims,back);
				return None;
			}
		}else if !dims.contains(&0){
			self.back=Some(Position::end(rank));
		}								// if both forward and reverse iteration have started, ensure we don't yield the same position from both ends
		if let Some(back)=self.back.as_deref_mut()&&let Some(front)=self.front.as_deref(){
			if greater_equals_position(dims,front,back){
										// if the iteration is done undo the decrement to make it not restart after the None
				increment_position(dims,back);
				return None
			}
		}

		self.back.clone()
	}
	fn nth_back(&mut self,n:usize)->Option<Self::Item>{
		let dims=self.layout.dims();
		let rank=dims.len();
									// if n is 0 return next_back early to skip the len calculation that takes as long as next_back's worst case
		if n==0{return self.next_back()}
		let len=self.len();
									// since this iterator fuses, we don't need to go farther than len. in fact, we don't need to go farther than len-1 as we can skip the decrement by using len to know when to end
		if len==0{return None}
		let k=n.min(len-1);
									// basically next_back but rewind by k+1 instead of increment
		if let Some(back)=self.back.as_deref_mut(){
			rewind_position(dims,k+1,back);
		}else if !dims.contains(&0){
			let mut back=Position::end(rank);
			rewind_position(dims,k,&mut back);

			self.back=Some(back);
		}

		if n>=len{return None}
		self.back.clone()
	}
}
impl ExactSizeIterator for PositionIntoIter{
	fn len(&self)->usize{self.state.len()}
}
impl ExactSizeIterator for PositionIter{
	fn len(&self)->usize{
		// The remaining length is computed by subtracting the current front position from the current back position using mixed-radix arithmetic.
		// Since the bounds are exclusive and may be absent before iteration begins, an additional most-significant radix-3 digit is introduced.
		// The additional radix-3 digit encodes the iterator state:
		// 0: before the first element (front==None)
		// 1: within the iteration (Some(position))
		// 2: after the last element (back==None)
		// This allows a None front to be encoded as 1 before the first item, and a None back to be encoded as 1 after the last item, so all iterator states may be handled by the same subtraction algorithm.
		// Because both stored bounds are exclusive to keep the stored positions in their existing allocations, the remaining length is back-front-1 rather than the usual half-open end-start
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
										// if the iteration has started, increment the front position. Otherwise, start from the first position. (unless dims contains 0 because empty tensors produce no positions)
		if let Some(front)=self.front.as_deref_mut(){
			if increment_position(dims,front)>0{
										// if it wraps the iteration is done and it's reached the end. undo the increment to make it not restart after the None.
				decrement_position(dims,front);
				return None;
			}
		}else if !dims.contains(&0){
			self.front=Some(Position::new(rank));
		}								// if both forward and reverse iteration have started, ensure we don't yield the same position from both ends
		if let Some(front)=self.front.as_deref_mut()&&let Some(back)=self.back.as_deref(){
			if greater_equals_position(dims,front,back){
										// if the iteration is done undo the increment to make it not restart after the None
				decrement_position(dims,front);
				return None
			}
		}

		self.front.clone()
	}
	fn nth(&mut self,n:usize)->Option<Self::Item>{
		let dims=self.layout.dims();
		let rank=dims.len();
									// if n is 0 return next early to skip the len calculation that takes as long as next's worst case
		if n==0{return self.next()}
		let len=self.len();
									// since this iterator fuses, we don't need to go farther than len. in fact, we don't need to go farther than len-1 as we can skip the decrement by using len to know when to end
		if len==0{return None}
		let k=n.min(len-1);
									// basically next but advance by k+1 instead of increment
		if let Some(front)=self.front.as_deref_mut(){
			advance_position(dims,k+1,front);
		}else if !dims.contains(&0){
			let mut front=Position::new(rank);
			advance_position(dims,k,&mut front);

			self.front=Some(front);
		}

		if n>=len{return None}
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
}
impl<P:SignedIndexPosition> PartialEq<(&[P],&Layout)> for Position{
	fn eq(&self,other:&(&[P],&Layout))->bool{*self==(other.0,&**other.1.dims())}
}
impl PartialEq for Position{
	fn eq(&self,other:&Self)->bool{self.partial_cmp(other)==Some(Ordering::Equal)}
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
								// If the positions have different ranks, we don't consider them able to can't be compared
		if qx.len()!=rank{return None}
								// If the positions have opposite signs, we can't compare without knowing the dims
		for ix in 0..rank{
			let (px,qx)=(px[ix],qx[ix]);
			if (px<0)^(qx<0){return None}
								// compare the position lexiconigraphically
			match px.cmp(&qx){
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
	/// returns current back position, which is usually the most recent position from next_back, but may be None or an exclusive bound on the yielded position if next_back has not been called
	pub fn back(&self)->Option<Position>{self.back.clone()}
	/// references the dimensions of the tensor whose position this iterates over
	pub fn dims(&self)->&[usize]{self.layout.dims()}
	/// returns current front position, which is usuallt the most recent position from next, but may be None or an exclusive bound on the yielded position if next_back has not been called
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
	#[track_caller]
	/// Advances this position by `distance` steps in last-axis-fastest iteration order. Panics if any coordinates at indices reached are out of bounds or if any dims at indices reached exceed isize::MAX, but may not reach all indices. Panics if dims and position have mismatched ranks
	/// Returns a carry value equal to the number of times the position has to wrap around the tensor before advancing the full distance.
	/// This return value behaves like the carry from mixed-radix addition, allowing positions over multiple groups of axes to be advanced independently.
	pub fn advance(&mut self,dims:&[usize],distance:usize)->usize{advance_position(dims,distance,self)}
	/// References the coordinates as a mutable slice.
	/// If the coordinates are shared with other `Position`s, they are cloned so that a unique slice may be returned
	pub fn as_mut_slice(&mut self)->&mut [isize]{Arc::make_mut(&mut self.0)}
	/// References the coordinates as a shared slice.
	pub fn as_slice(&self)->&[isize]{self.0.as_ref()}
	/// Reinterprets the shared coordinates as `usize`. This performs a bitwise reinterpretation of each coordinate and does not normalize negative indices. It is recommended that callers ensure all coordinates are positive before calling. For example, `-1` becomes `usize::MAX`, which is unlikely to be what you want.
	/// This method is primarily intended for low-level code that treats the coordinates as raw integers.
	pub fn cast_unsigned(&self)->&[usize]{
		let ix:&[isize]=self.as_slice();
		unsafe{		// safety: isize and usize have identical size and alignment, and the returned slice has the same lifetime, ptr, and len as the original.
			slice::from_raw_parts(ix.as_ptr() as *const usize,ix.len())
		}
	}
	/// Rewind this position by 1 step in last-axis-fastest iteration order. Equivalent to rewind(dims,1), but optimized for a single step.
	pub fn decrement(&mut self,dims:&[usize])->usize{decrement_position(dims,self)}
	/// Creates the last position of a tensor with the given rank. The coordinates are all initialized to -1.
	/// This represents the last component along every axis when interpreted using signed indexing.
	pub fn end(rank:usize)->Self{Self(vec![-1;rank].into())}
	/// Advance this position by 1 step in last-axis-fastest iteration order. Equivalent to advance(dims,1), but optimized for a single step.
	pub fn increment(&mut self,dims:&[usize])->usize{increment_position(dims,self)}
	/// Creates the first position of a tensor with the given rank. The coordinates are all initialized to 0
	pub fn new(rank:usize)->Self{Self(vec![0;rank].into())}
	/// /// Returns the rank of this position. position.rank() is consistent with this library's vocabulary than using position.len() through deref
	pub fn rank(&self)->usize{self.len()}
	#[track_caller]
	/// Rewinds this position by `distance` steps in last-axis-fastest iteration order. Panics if any coordinates at indices reached are out of bounds or if any dims at indices reached exceed isize::MAX, but may not reach all indices. Panics if dims and position have mismatched ranks
	/// Returns a carry value equal to the number of times the position has to wrap around the tensor before rewinding the full distance.
	/// This return value behaves like the carry from mixed-radix subtraction, allowing positions over multiple groups of axes to be rewound independently.
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
	/// Convert the indices to their unsigned forms. Returns None if out of bounds, but may still modify some of the coordinates. Check bounds first if coordinates must be unchanged in the out of bounds case
	pub fn unsign(&mut self,dims:&[usize])->Option<&[usize]>{
		assert_eq!(dims.len(),self.len());

		for (&dim,index) in dims.iter().zip(self.iter_mut()){*index=unsign_position(dim,*index)? as isize}
		Some(self.cast_unsigned())
	}
}

#[cfg(test)]
mod tests{
	#[test]
	fn advance(){
		let mut dims=vec![1,1];
		let mut position:Vec<isize>=vec![0,0];

		assert_eq!(advance_position(&dims,1,&mut position),1);
		assert_eq!(position,[0,0]);

		dims=vec![1,2];

		assert_eq!(advance_position(&dims,1,&mut position),0);
		assert_eq!(position,[0,1]);
		assert_eq!(advance_position(&dims,1,&mut position),1);
		assert_eq!(position,[0,0]);

		dims=vec![2,4];

		assert_eq!(advance_position(&dims,18,&mut position),2);
		assert_eq!(position,[0,2]);
	}
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
	fn position_iter_nth(){
		let iter=PositionIter::new([4,1,3,2,9]);
		for n in 0..12*18+10{
			let mut a = iter.clone();
			let mut b = iter.clone();

			dbg!(n);
			assert_eq!(a.nth(n), {
				for _ in 0..n {
					b.next();
				}
				b.next()
			});

			assert_eq!(a.len(),iter.len().saturating_sub(n+1));
			assert_eq!(b.len(),iter.len().saturating_sub(n+1));
		}
	}
	#[test]
	fn position_iter_nth_back(){
		let iter=PositionIter::new([4,1,3,2,9]);
		for n in 0..12*18+10{
			let mut a = iter.clone();
			let mut b = iter.clone();

			assert_eq!(a.nth_back(n), {
				for _ in 0..n {
					b.next_back();
				}
				b.next_back()
			});

			assert_eq!(a.len(),iter.len().saturating_sub(n+1));
			assert_eq!(b.len(),iter.len().saturating_sub(n+1));
		}
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
/// Advances this position by `distance` steps in last-axis-fastest iteration order. Panics if any coordinates at indices reached are out of bounds or if any dims at indices visited exceed isize::MAX. Panics if dims and position have mismatched ranks. Also panics if coordinates fail to convert between the canonical isize and their stored types.
/// Returns a carry value equal to the number of times the position has to wrap around the tensor before advancing the full distance.
/// This return value behaves like the carry from mixed-radix addition, allowing positions over multiple groups of axes to be advanced independently.
pub fn advance_position<P:SignedIndexPosition>(dims:&[usize],mut distance:usize,position:&mut [P])->usize{
	assert_eq!(dims.len(),position.len());
													// this algorithm is conceptually similar to mixed-radix addition on the coordinates,
	for (&dim,coordinate) in dims.iter().rev().zip(position.iter_mut().rev()){
		if distance==0{break}
		assert!(dim<=isize::MAX as usize);
													// get coordinate and check bounds. as dim==0 case is already covered by bounds check, division by 0 problems won't occur later.
		let mut px=coordinate.expect_isize("coordinates must fit in isize");
		if !(-(dim as isize)..dim as isize).contains(&px){panic!("coordinate {px} is out of bounds for dim {dim}")};
													// we know at least distance/dim carries will be needed, but if adding distance%dim to the coordinate wraps around, we'll need one more. Negative coordinates wrap when they reach 0, positive coordinates wrap when they reach dim.
		let carrybound=if px<0{0}else{dim as isize};
		let dx=distance%dim;

		px+=dx as isize;
													// correct position if it exceeded the carry bound
		let carry=px>=carrybound;
		if carry{px-=dim as isize}
													// set coordinate to the updated value and update distance to be the distance to advance along the next axis
		*coordinate=px.expect_coordinate("updated coordinates must fit in their original type");
		distance=carry as usize+distance/dim;
	}
	distance
}
/// computes the length of a buffer required to hold a tensor with the given dims and strides. This expects an equal number of dims and strides, and that no dim exceeds isize::MAX, and that the buffer len doesn't exceed isize::MAX, but it doesn't check. Use error::checked_len to check the dims for validity while computing the length. If any dim is 0, the result is 0. Otherwise, each axis independently contributes (dim-1)*|stride| to the maximum offset. The required buffer length is 1 more than the maximum offset
pub fn buffer_len(dims:&[usize],strides:&[isize])->usize{
	if dims.contains(&0){return 0}
	dims.iter().rev().zip(strides.iter().rev()).fold(1,|acc,(&dim,&stride)|acc+(dim-1)*stride.abs() as usize)
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
	assert_eq!(dims.len(),position.len());
	let mut distance=1;
													// this algorithm is conceptually similar to mixed-radix subtraction on the coordinates,
	for (&dim,coordinate) in dims.iter().rev().zip(position.iter_mut().rev()){
		if distance==0{break}
		assert!(dim<=isize::MAX as usize);
													// get coordinate and check bounds. as dim==0 case is already covered by bounds check, division by 0 problems won't occur later.
		let mut px=coordinate.expect_isize("coordinates must fit in isize");
		if !(-(dim as isize)..dim as isize).contains(&px){panic!("coordinate {px} is out of bounds for dim {dim}")};

		let carrybound=if px<0{-(dim as isize)}else{0};
		px-=1;
													// correct position if it exceeded the carry bound
		let carry=px<carrybound;
		if carry{px+=dim as isize}
													// set coordinate to the updated value and update distance to be the distance to rewind along the next axis
		*coordinate=px.expect_coordinate("updated coordinates must fit in their original type");
		distance=if carry{1}else{0};
	}
	distance
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
/// Same as advance_position(dims,1,position), but optimized to reduce division. see advance_position for more details
pub fn increment_position<P:SignedIndexPosition>(dims:&[usize],position:&mut [P])->usize{
	assert_eq!(dims.len(),position.len());
	let mut distance=1;
													// this algorithm is conceptually similar to mixed-radix addition on the coordinates,
	for (&dim,coordinate) in dims.iter().rev().zip(position.iter_mut().rev()){
		if distance==0{break}
		assert!(dim<=isize::MAX as usize);
													// get coordinate and check bounds. as dim==0 case is already covered by bounds check, division by 0 problems won't occur later.
		let mut px=coordinate.expect_isize("coordinates must fit in isize");
		if !(-(dim as isize)..dim as isize).contains(&px){panic!("coordinate {px} is out of bounds for dim {dim}")};

		let carrybound=if px<0{0}else{dim as isize};
		px+=1;
													// correct position if it exceeded the carry bound
		let carry=px>=carrybound;
		if carry{px-=dim as isize}
													// set coordinate to the updated value and update distance to be the distance to advance along the next axis
		*coordinate=px.expect_coordinate("updated coordinates must fit in their original type");
		distance=if carry{1}else{0};
	}
	distance
}
#[track_caller]
/// compare if two positions refer to the same component.
pub fn less_equals_position<P:SignedIndexPosition,Q:SignedIndexPosition>(dims:&[usize],px:&[P],qx:&[Q])->bool{compare_position(dims,px,qx)!=Ordering::Greater}
#[track_caller]
/// compare if two positions refer to the same component.
pub fn less_position<P:SignedIndexPosition,Q:SignedIndexPosition>(dims:&[usize],px:&[P],qx:&[Q])->bool{compare_position(dims,px,qx)==Ordering::Less}
#[track_caller]
/// Rewinds this position by `distance` steps in last-axis-fastest iteration order. Panics if any coordinates at indices reached are out of bounds or if any dims at indices visited exceed isize::MAX. Panics if dims and position have mismatched ranks. Also panics if coordinates fail to convert between the canonical isize and their stored types.
/// Returns a carry value equal to the number of times the position has to wrap around the tensor before rewinding the full distance.
/// This return value behaves like the carry from mixed-radix addition, allowing positions over multiple groups of axes to be rewound independently.
pub fn rewind_position<P:SignedIndexPosition>(dims:&[usize],mut distance:usize,position:&mut [P])->usize{
	assert_eq!(dims.len(),position.len());
													// this algorithm is conceptually similar to mixed-radix subtraction on the coordinates,
	for (&dim,coordinate) in dims.iter().rev().zip(position.iter_mut().rev()){
		if distance==0{break}
		assert!(dim<=isize::MAX as usize);
													// get coordinate and check bounds. as dim==0 case is already covered by bounds check, division by 0 problems won't occur later.
		let mut px=coordinate.expect_isize("coordinates must fit in isize");
		if !(-(dim as isize)..dim as isize).contains(&px){panic!("coordinate {px} is out of bounds for dim {dim}")};
													// we know at least distance/dim carries will be needed, but if subtracting distance%dim to the coordinate wraps around, we'll need one more. Negative coordinates wrap when they fall below -dim, positive coordinates wrap when they fall below 0.
		let carrybound=if px<0{-(dim as isize)}else{0};
		let dx=distance%dim;

		px-=dx as isize;
													// correct position if it exceeded the carry bound
		let carry=px<carrybound;
		if carry{px+=dim as isize}
													// set coordinate to the updated value and update distance to be the distance to rewind along the next axis
		*coordinate=px.expect_coordinate("updated coordinates must fit in their original type");
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
/// Iterates over positions in a tensor, in row-major (last-axis-fastest) order.
/// `front` and `back`, when present, are always valid in-bounds positions. Exhaustion is represented by iterator state rather than by storing out-of-bounds sentinel positions.
pub struct PositionIter{layout:Layout,front:Option<Position>,back:Option<Position>}
#[derive(Debug,Default)]
#[cfg_attr(feature="serial",derive(Deserialize,Serialize))]
#[repr(transparent)]
/// A postion within a tensor. Coordinates are signed and stored in axis order. Cloning is inexpensive because its coordinates are stored in shared reference counted memory, however, mutating a position may require a more expensive cloning of the inner data (clone-on-write semantics).
/// A Position contains one coordinate for each axis of a tensor. Positions use signed coordinates so that negative indexing may be represented prior to normalization.
/// A Position does not include the tensor's dimensions. Consequently, two positions cannot always be compared for equality of the referenced component without also knowing the tensor's dimensions, since negative coordinates may normalize to different values depending on the dimensions.
/// Still, PartialOrd<Position> for Position is implemented, returning None if the positions cannot be compared. (due to having opposite signs at any index, or having different ranks)
/// For example Position::from([-1]).partial_cmp(Position::from([3])) is None, because we don't know whether the dim is 4 (making them equal) or some other value (making them unequal)
/// However, Position::from([-1])==Position::from([-1]) is true, because those two positions would refer to the same component regardless of the dimension of axis 0.
pub struct Position(Arc<[isize]>);
#[derive(Clone,Debug,Default)]
/// Iterator over Position, yielding its coordinates in axis order.
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
