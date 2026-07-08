impl<E:Default,S:AsRef<Path>> LoadSheet<S> for IOResult<Tensor<E>> where Tensor<E>:for<'a> ReadSheet<&'a Spreadsheet>{
	fn load_sheet(sheetpath:S)->Self{
		let mut result=Ok(Default::default());

		result.read_sheet(&mut [0,0,0],sheetpath.as_ref());
		result
	}
}
impl<E:Default,S:AsRef<Path>> LoadSheet<S> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Path>{
	fn load_sheet(sheetpath:S)->Self{Self::from_sheet(sheetpath.as_ref())}
}
impl<E:Default> From<Spreadsheet> for Tensor<E> where Tensor<E>:ReadSheet<Spreadsheet>{
	fn from(sheet:Spreadsheet)->Tensor<E>{Self::from_sheet(sheet)}
}
impl<E:Default> From<Worksheet> for Tensor<E> where Tensor<E>:ReadSheet<Worksheet>{
	fn from(sheet:Worksheet)->Tensor<E>{Self::from_sheet(sheet)}
}
impl<E:Default> ReadSheet<&Path> for IOResult<Tensor<E>> where Tensor<E>:for<'a> ReadSheet<&'a Spreadsheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Path){
		let x=match xlsx::read(sheet){Err(e)=>return *self=Err(IOError::new(IOErrorKind::Other,e)),Ok(x)=>x};
		if let Ok(t)=self{return t.read_sheet(indices,&x)}

		let mut t:Tensor<E>=Default::default();
		t.read_sheet(indices,&x);

		*self=Ok(t)
	}
}
impl<E:Default> ReadSheet<&Path> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Spreadsheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Path){
		if self.dims().len()==0&&self.buffer_len()==0{
			self.dims_mut().push(0);
			self.strides_mut().push(1);
		}
		let x=xlsx::read(sheet).unwrap();
		self.read_sheet(indices,&x);
	}
}
impl<E:Default> ReadSheet<&Spreadsheet> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Worksheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Spreadsheet){
		if self.dims().len()==0&&self.buffer_len()==0{
			self.dims_mut().push(0);
			self.strides_mut().push(1);
		}
		assert!(indices.len()>=3);
		let im3=indices.len()-3;

		for n in 0..{
			indices[im3]=n as isize;
			self.read_sheet(indices,if let Some(s)=sheet.get_sheet(&n){s}else{break});
		}
	}
}
impl<E:Default> ReadSheet<Spreadsheet> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Spreadsheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:Spreadsheet){self.read_sheet(indices,&sheet)}
}
impl<E:Default> ReadSheet<Worksheet> for Tensor<E> where Tensor<E>:for<'a> ReadSheet<&'a Worksheet>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:Worksheet){self.read_sheet(indices,&sheet)}
}
impl ReadSheet<&Worksheet> for Tensor<Option<Cell>>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Worksheet){
		if self.dims().len()==0&&self.buffer_len()==0{
			self.dims_mut().push(0);
			self.strides_mut().push(1);
		}
		assert!(indices.len()>=2);

		let (xstop,ystop)=sheet.get_highest_column_and_row();
		let im2=indices.len()-2;

		while indices.len()>self.rank(){
			self.unsqueeze_dim(0);
		}

		self.pad_dim(-2,xstop as usize,None);
		self.pad_dim(-1,ystop as usize,None);
		if im2>0{
			self.pad_dim(-3,(indices[im2-1]+1) as usize,None);
		}

		for x in 0..xstop{
			indices[im2]=x as isize;
			for y in 0..ystop{
				indices[im2+1]=y as isize;
				self[&indices[..]]=sheet.get_cell((x+1,y+1)).cloned();
			}
		}
	}
}
impl ReadSheet<&Worksheet> for Tensor<String>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Worksheet){
		if self.dims().len()==0&&self.buffer_len()==0{
			self.dims_mut().push(0);
			self.strides_mut().push(1);
		}
		assert!(indices.len()>=2);

		let (xstop,ystop)=sheet.get_highest_column_and_row();
		let im2=indices.len()-2;

		while indices.len()>self.rank(){
			self.unsqueeze_dim(0);
		}

		self.pad_dim(-2,xstop as usize,String::new());
		self.pad_dim(-1,ystop as usize,String::new());
		if im2>0{
			self.pad_dim(-3,(indices[im2-1]+1) as usize,String::new());
		}

		for x in 0..xstop{
			indices[im2]=x as isize;
			for y in 0..ystop{
				indices[im2+1]=y as isize;
				self[&indices[..]]=sheet.get_value((x+1,y+1));
			}
		}
	}
}
impl ReadSheet<&Worksheet> for Tensor<f32>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Worksheet){
		if self.dims().len()==0&&self.buffer_len()==0{
			self.dims_mut().push(0);
			self.strides_mut().push(1);
		}
		assert!(indices.len()>=2);

		let (xstop,ystop)=sheet.get_highest_column_and_row();
		let im2=indices.len()-2;

		while indices.len()>self.rank(){
			self.unsqueeze_dim(0);
		}

		self.pad_dim(-2,xstop as usize,f32::NAN);
		self.pad_dim(-1,ystop as usize,f32::NAN);
		if im2>0{
			self.pad_dim(-3,(indices[im2-1]+1) as usize,f32::NAN);
		}

		for x in 0..xstop{
			indices[im2]=x as isize;
			for y in 0..ystop{
				indices[im2+1]=y as isize;
				self[&indices[..]]=sheet.get_value_number((x+1,y+1)).unwrap_or(f64::NAN) as f32;
			}
		}
	}
}
impl ReadSheet<&Worksheet> for Tensor<f64>{
	fn read_sheet(&mut self,indices:&mut [isize],sheet:&Worksheet){
		if self.dims().len()==0&&self.buffer_len()==0{
			self.dims_mut().push(0);
			self.strides_mut().push(1);
		}
		assert!(indices.len()>=2);

		let (xstop,ystop)=sheet.get_highest_column_and_row();
		let im2=indices.len()-2;

		while indices.len()>self.rank(){
			self.unsqueeze_dim(0);
		}

		self.pad_dim(-2,xstop as usize,f64::NAN);
		self.pad_dim(-1,ystop as usize,f64::NAN);
		if im2>0{self.pad_dim(-3,(indices[im2-1]+1) as usize,f64::NAN)}

		for x in 0..xstop{
			indices[im2]=x as isize;
			for y in 0..ystop{
				indices[im2+1]=y as isize;
				self[&indices[..]]=sheet.get_value_number((x+1,y+1)).unwrap_or(f64::NAN);
			}
		}
	}
}
#[cfg(test)]
mod tests{
	#[test]
	fn find_table(){
		use crate::match_tensor::{CellPattern,self};

		let expected=vec![
			"Single Investment Property",	"",						"",			"",						"",
			"Maximum LTV/CLTV",				"",						">= 1.00",	"",						"",
			"Minimum Credit Score",			"Maximum Loan Amount",	"Purchase",	"Rate/Term Refinance",	"Cash-Out Refinance",
			"700",							"1000000",				"80",		"80",					"80*",
			"",								"1500000",				"80",		"75",					"75",
			"",								"2000000",				"75",		"70",					"70",
			"",								"3000000",				"70",		"65",					"65",
			"",								"3500000",				"70",		"65",					"NA",
			"680",							"1000000",				"75",		"75",					"75",
			"",								"1500000",				"75",		"75",					"75",
			"",								"2000000",				"75",		"75",					"75",
			"",								"2500000",				"70",		"65",					"65",
			"",								"3000000",				"65",		"NA",					"NA",
			"640",							"1000000",				"75",		"70",					"70",
			"",								"1500000",				"70",		"70",					"70",
			"",								"2000000",				"65",		"NA",					"NA",
			"",								"3000000",				"60",		"NA",					"NA",
		];
		let matrixfile="Matrix November 10th FINAL.xlsx";
		let pattern=vec![
			CellPattern::Text("Single Investment Property".into()),CellPattern::Anything,CellPattern::Anything,CellPattern::Anything,CellPattern::Anything,
			CellPattern::Text("Maximum LTV".into()),CellPattern::Anything,CellPattern::Text(">=1.0".into()),CellPattern::Anything,CellPattern::Anything,
			CellPattern::Text("Minimum credit score".into()),CellPattern::Text("Maximum Loan Amount".into()),CellPattern::Text("Purchase".into()),CellPattern::Text("Rate/Term Refinance".into()),CellPattern::Text("Cash-Out Refinance".into()),
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
			CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,CellPattern::Numeric,
		];

		let mut data:Tensor<String>=Tensor::load_sheet(matrixfile);
		let mut pattern:Tensor<CellPattern>=Tensor::from(pattern);

		data.slice(&[0..3,0..5,0..30]);
		pattern.reshape([17,5]);
		pattern.swap_dims(0,1);

		let (table, _cost)=match_tensor::grab_table_default(data,pattern);

		assert_eq!(table.view_ref().swap_dims(-1,-2).to_flat_vec(None),expected);
	}
	#[test]
	fn read_matrix(){
		let matrixfile="Matrix November 10th FINAL.xlsx";
		let numbers:Tensor<f32>=Tensor::load_sheet(matrixfile);
		let text:Tensor<String>=Tensor::load_sheet(matrixfile);

		assert!(numbers[[0,0,0]].is_nan());
		assert!(numbers[[0,0,1]].is_nan());
		assert!(numbers[[0,1,0]].is_nan());
		assert_eq!(numbers.into_view().slice([1..2,2..4,3..8]).swap_dims(-1,-2).into_tensor().into_flat_vec(None),vec![80.0,80.0,80.0,75.0,75.0,70.0,70.0,65.0,70.0,65.0]);
		assert_eq!(text[[0,0,1]],"test");
	}
	use super::*;
}
/// provide function for loading spreadsheet to tensor by path refs
pub trait LoadSheet<S>{
	/// loads a tensor from the spreadsheet file
	fn load_sheet(sheetpath:S)->Self;
}
/// how to read the columns of a spreadsheet into a tensor. dims: [sheets, columns, rows]
pub trait ReadSheet<S>{
	/// reads the excel sheet into a tensor
	fn from_sheet(sheet:S)->Self where Self:Default{
		let mut tensor=Self::default();

		tensor.read_sheet(&mut [0,0,0],sheet);
		tensor
	}
	/// reads the tensor from excel format
	fn read_sheet(&mut self,indices:&mut [isize],sheet:S);
}
use crate::builtin_tensor::Tensor;
use std::{
	io::{Error as IOError,ErrorKind as IOErrorKind,Result as IOResult},path::Path
};
use umya_spreadsheet::{Cell,Spreadsheet,Worksheet,reader::xlsx};
