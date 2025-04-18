// Copyright 2024-2025 Irreducible Inc.

use std::{
	cmp::Ordering,
	fmt::{self, Display},
	iter::{Product, Sum},
	ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},

};

use binius_field::{Field, PackedField, TowerField};
use binius_macros::{DeserializeBytes, SerializeBytes};

use super::error::Error;


/// A tiny “dual” number carrying
///  - `real`: the function value
///  - `derivs`: the length‑n vector of partials ∂f/∂xᵢ
#[derive(Clone, Debug)]
pub struct Dual<F: Field> {
    pub real:   F,
    pub derivs: Vec<F>,
}

impl<F: Field> Dual<F> {
    /// constant c ⇒ (real=c, derivs=0⃗)
    pub fn constant(c: F, n: usize) -> Self {
        Self { real: c, derivs: vec![F::ZERO; n] }
    }
    /// variable xᵢ ⇒ (real=0, derivs=eᵢ)
    pub fn variable(i: usize, n: usize) -> Self {
        let mut d = vec![F::ZERO; n];
        d[i] = F::ONE;
        Self { real: F::ZERO, derivs: d }
    }

    /// exponentiate by a non‑negative integer,
    /// error if exp>1 and there’s any derivative
    pub fn pow_uint(self, exp: usize) -> Result<Self, Error> {
        match exp {
            0 => Ok(Self::constant(F::ONE, self.derivs.len())),
            1 => Ok(self),
            _ => {
                // base must be constant to stay linear
                if self.derivs_constant() {
                    Ok(Self {
                        real:   self.real.pow(exp as u64),
                        derivs: vec![F::ZERO; self.derivs.len()],
                    })
                } else {
                    Err(Error::NonLinearExpression)
                }
            }
        }
    }

	pub fn derivs_constant(&self) -> bool {
		self.derivs.iter().all(|&d| d == F::ZERO)
	}

    /// multiply, error if both sides carry a derivative
    pub fn checked_mul(self, rhs: Self) -> Result<Self, Error> {
        let n = self.derivs.len();
        let left_const  = self.derivs_constant();
        let right_const = rhs.derivs_constant();

        let real = self.real * rhs.real;
        let derivs = match (left_const, right_const) {
            (true, true)   => vec![F::ZERO; n],
            (true, false)  => rhs.derivs.iter().map(|&d| self.real * d).collect(),
            (false, true)  => self.derivs.iter().map(|&d| d * rhs.real).collect(),
            (false, false) => return Err(Error::NonLinearExpression),
        };

        Ok(Self { real, derivs })
    }
}

// We can still use standard `+` for addition:
impl<F: Field> Add for Dual<F> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let derivs = self.derivs
            .into_iter()
            .zip(other.derivs)
            .map(|(a,b)| a + b)
            .collect();
        Self {
            real:   self.real + other.real,
            derivs,
        }
    }
}

impl<F: Field> From<F> for Dual<F> {
	fn from(value: F) -> Self {
		Self {
			real: value,
			derivs: Vec::new(),
		}
	}
}

impl<F: Field> From<Dual<F>> for LinearNormalForm<F> {
	fn from(value: Dual<F>) -> Self {
		Self {
			constant: value.real,
			var_coeffs: value.derivs,
		}
	}
}


/// Arithmetic expressions that can be evaluated symbolically.
///
/// Arithmetic expressions are trees, where the leaves are either constants or variables, and the
/// non-leaf nodes are arithmetic operations, such as addition, multiplication, etc. They are
/// specific representations of multivariate polynomials.
#[derive(Debug, Clone, PartialEq, Eq, SerializeBytes, DeserializeBytes)]
pub enum ArithExpr<F: Field> {
	Const(F),
	Var(usize),
	Add(Box<ArithExpr<F>>, Box<ArithExpr<F>>),
	Mul(Box<ArithExpr<F>>, Box<ArithExpr<F>>),
	Pow(Box<ArithExpr<F>>, u64),
}

impl<F: Field + Display> Display for ArithExpr<F> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Const(v) => write!(f, "{v}"),
			Self::Var(i) => write!(f, "x{i}"),
			Self::Add(x, y) => write!(f, "({} + {})", &**x, &**y),
			Self::Mul(x, y) => write!(f, "({} * {})", &**x, &**y),
			Self::Pow(x, p) => write!(f, "({})^{p}", &**x),
		}
	}
}

impl<F: Field> ArithExpr<F> {
	/// The number of variables the expression contains.
	pub fn n_vars(&self) -> usize {
		match self {
			Self::Const(_) => 0,
			Self::Var(index) => *index + 1,
			Self::Add(left, right) | Self::Mul(left, right) => left.n_vars().max(right.n_vars()),
			Self::Pow(id, _) => id.n_vars(),
		}
	}

	/// The total degree of the polynomial the expression represents.
	pub fn degree(&self) -> usize {
		match self {
			Self::Const(_) => 0,
			Self::Var(_) => 1,
			Self::Add(left, right) => left.degree().max(right.degree()),
			Self::Mul(left, right) => left.degree() + right.degree(),
			Self::Pow(base, exp) => base.degree() * *exp as usize,
		}
	}

	/// Return a new arithmetic expression that contains only the terms of highest degree
	/// (useful for interpolation at Karatsuba infinity point).
	pub fn leading_term(&self) -> Self {
		let (_, expr) = self.leading_term_with_degree();
		expr
	}

	/// Same as `leading_term`, but returns the total degree as the first tuple element as well.
	pub fn leading_term_with_degree(&self) -> (usize, Self) {
		match self {
			expr @ Self::Const(_) => (0, expr.clone()),
			expr @ Self::Var(_) => (1, expr.clone()),
			Self::Add(left, right) => {
				let (lhs_degree, lhs) = left.leading_term_with_degree();
				let (rhs_degree, rhs) = right.leading_term_with_degree();
				match lhs_degree.cmp(&rhs_degree) {
					Ordering::Less => (rhs_degree, rhs),
					Ordering::Equal => (lhs_degree, Self::Add(Box::new(lhs), Box::new(rhs))),
					Ordering::Greater => (lhs_degree, lhs),
				}
			}
			Self::Mul(left, right) => {
				let (lhs_degree, lhs) = left.leading_term_with_degree();
				let (rhs_degree, rhs) = right.leading_term_with_degree();
				(lhs_degree + rhs_degree, Self::Mul(Box::new(lhs), Box::new(rhs)))
			}
			Self::Pow(base, exp) => {
				let (base_degree, base) = base.leading_term_with_degree();
				(base_degree * *exp as usize, Self::Pow(Box::new(base), *exp))
			}
		}
	}

	pub fn pow(self, exp: u64) -> Self {
		Self::Pow(Box::new(self), exp)
	}

	pub const fn zero() -> Self {
		Self::Const(F::ZERO)
	}

	pub const fn one() -> Self {
		Self::Const(F::ONE)
	}

	/// Creates a new expression with the variable indices remapped.
	///
	/// This recursively replaces the variable sub-expressions with an index `i` with the variable
	/// `indices[i]`.
	///
	/// ## Throws
	///
	/// * [`Error::IncorrectArgumentLength`] if indices has length less than the current number of
	///   variables
	pub fn remap_vars(self, indices: &[usize]) -> Result<Self, Error> {
		let expr = match self {
			Self::Const(_) => self,
			Self::Var(index) => {
				let new_index =
					indices
						.get(index)
						.ok_or_else(|| Error::IncorrectArgumentLength {
							arg: "subset".to_string(),
							expected: index,
						})?;
				Self::Var(*new_index)
			}
			Self::Add(left, right) => {
				let new_left = left.remap_vars(indices)?;
				let new_right = right.remap_vars(indices)?;
				Self::Add(Box::new(new_left), Box::new(new_right))
			}
			Self::Mul(left, right) => {
				let new_left = left.remap_vars(indices)?;
				let new_right = right.remap_vars(indices)?;
				Self::Mul(Box::new(new_left), Box::new(new_right))
			}
			Self::Pow(base, exp) => {
				let new_base = base.remap_vars(indices)?;
				Self::Pow(Box::new(new_base), exp)
			}
		};
		Ok(expr)
	}

	/// Substitute variable with index `var` with a constant `value`
	pub fn const_subst(self, var: usize, value: F) -> Self {
		match self {
			Self::Const(_) => self,
			Self::Var(index) => {
				if index == var {
					Self::Const(value)
				} else {
					self
				}
			}
			Self::Add(left, right) => {
				let new_left = left.const_subst(var, value);
				let new_right = right.const_subst(var, value);
				Self::Add(Box::new(new_left), Box::new(new_right))
			}
			Self::Mul(left, right) => {
				let new_left = left.const_subst(var, value);
				let new_right = right.const_subst(var, value);
				Self::Mul(Box::new(new_left), Box::new(new_right))
			}
			Self::Pow(base, exp) => {
				let new_base = base.const_subst(var, value);
				Self::Pow(Box::new(new_base), exp)
			}
		}
	}

	pub fn convert_field<FTgt: Field + From<F>>(&self) -> ArithExpr<FTgt> {
		match self {
			Self::Const(val) => ArithExpr::Const((*val).into()),
			Self::Var(index) => ArithExpr::Var(*index),
			Self::Add(left, right) => {
				let new_left = left.convert_field();
				let new_right = right.convert_field();
				ArithExpr::Add(Box::new(new_left), Box::new(new_right))
			}
			Self::Mul(left, right) => {
				let new_left = left.convert_field();
				let new_right = right.convert_field();
				ArithExpr::Mul(Box::new(new_left), Box::new(new_right))
			}
			Self::Pow(base, exp) => {
				let new_base = base.convert_field();
				ArithExpr::Pow(Box::new(new_base), *exp)
			}
		}
	}

	pub fn try_convert_field<FTgt: Field + TryFrom<F>>(
		&self,
	) -> Result<ArithExpr<FTgt>, <FTgt as TryFrom<F>>::Error> {
		Ok(match self {
			Self::Const(val) => ArithExpr::Const(FTgt::try_from(*val)?),
			Self::Var(index) => ArithExpr::Var(*index),
			Self::Add(left, right) => {
				let new_left = left.try_convert_field()?;
				let new_right = right.try_convert_field()?;
				ArithExpr::Add(Box::new(new_left), Box::new(new_right))
			}
			Self::Mul(left, right) => {
				let new_left = left.try_convert_field()?;
				let new_right = right.try_convert_field()?;
				ArithExpr::Mul(Box::new(new_left), Box::new(new_right))
			}
			Self::Pow(base, exp) => {
				let new_base = base.try_convert_field()?;
				ArithExpr::Pow(Box::new(new_base), *exp)
			}
		})
	}

	/// Whether expression is a composite node, and not a leaf.
	pub const fn is_composite(&self) -> bool {
		match self {
			Self::Const(_) | Self::Var(_) => false,
			Self::Add(_, _) | Self::Mul(_, _) | Self::Pow(_, _) => true,
		}
	}

	/// Returns `Some(F)` if the expression is a constant.
	pub const fn constant(&self) -> Option<F> {
		match self {
			Self::Const(value) => Some(*value),
			_ => None,
		}
	}

	/// Creates a new optimized expression.
	///
	/// Recursively rewrites the expression for better evaluation performance. Performs constant folding,
	/// as well as leverages simple rewriting rules around additive/multiplicative identities and addition
	/// in characteristic 2.
	pub fn optimize(&self) -> Self {
		match self {
			Self::Const(_) | Self::Var(_) => self.clone(),
			Self::Add(left, right) => {
				let left = left.optimize();
				let right = right.optimize();
				match (left, right) {
					// constant folding
					(Self::Const(left), Self::Const(right)) => Self::Const(left + right),
					// 0 + a = a + 0 = a
					(Self::Const(left), right) if left == F::ZERO => right,
					(left, Self::Const(right)) if right == F::ZERO => left,
					// a + a = 0 in char 2
					// REVIEW: relies on precise structural equality, find a better way
					(left, right) if left == right && F::CHARACTERISTIC == 2 => {
						Self::Const(F::ZERO)
					}
					// fallback
					(left, right) => Self::Add(Box::new(left), Box::new(right)),
				}
			}
			Self::Mul(left, right) => {
				let left = left.optimize();
				let right = right.optimize();
				match (left, right) {
					// constant folding
					(Self::Const(left), Self::Const(right)) => Self::Const(left * right),
					// 0 * a = a * 0 = 0
					(left, right)
						if left == Self::Const(F::ZERO) || right == Self::Const(F::ZERO) =>
					{
						Self::Const(F::ZERO)
					}
					// 1 * a = a * 1 = a
					(Self::Const(left), right) if left == F::ONE => right,
					(left, Self::Const(right)) if right == F::ONE => left,
					// fallback
					(left, right) => Self::Mul(Box::new(left), Box::new(right)),
				}
			}
			Self::Pow(id, exp) => {
				let id = id.optimize();
				match id {
					Self::Const(value) => Self::Const(PackedField::pow(value, *exp)),
					Self::Pow(id_inner, exp_inner) => Self::Pow(id_inner, *exp * exp_inner),
					id => Self::Pow(Box::new(id), *exp),
				}
			}
		}
	}

	// pub fn linear_normal_form(&self) -> Result<LinearNormalForm<F>, Error> {
	// 	self.sparse_linear_normal_form().map(Into::into)


	// struct SparseLinearNormalForm<F: Field> {
	// 	/// The constant offset of the expression.
	// 	pub constant: F,
	// 	/// A map of variable indices to their coefficients.
	// 	pub var_coeffs: HashMap<usize, F>,
	// 	/// The `var_coeffs` vector len if converted to [`LinearNormalForm`].
	// 	/// It is used for optimization of conversion to [`LinearNormalForm`].
	// 	pub dense_linear_form_len: usize,
	// }

	// fn sparse_linear_normal_form(&self) -> Result<SparseLinearNormalForm<F>, Error> {
	// 	match self {
	// 		Self::Const(val) => Ok((*val).into()),
	// 		Self::Var(index) => Ok(SparseLinearNormalForm {
	// 			constant: F::ZERO,
	// 			dense_linear_form_len: *index + 1,
	// 			var_coeffs: [(*index, F::ONE)].into(),
	// 		}),
	// 		Self::Add(left, right) => {
	// 			Ok(left.sparse_linear_normal_form()? + right.sparse_linear_normal_form()?)
	// 		}
	// 		Self::Mul(left, right) => {
	// 			left.sparse_linear_normal_form()? * right.sparse_linear_normal_form()?
	// 		}
	// 		Self::Pow(_, 0) => Ok(F::ONE.into()),
	// 		Self::Pow(expr, 1) => expr.sparse_linear_normal_form(),
	// 		Self::Pow(expr, pow) => expr.sparse_linear_normal_form().and_then(|linear_form| {
	// 			if linear_form.dense_linear_form_len != 0 {
	// 				return Err(Error::NonLinearExpression);
	// 			}
	// 			Ok(linear_form.constant.pow(*pow).into())
	// 		}),
	// 	}
	// }

	/// Returns the normal form of an expression if it is linear.
	///
	/// ## Throws
	///
	/// - [`Error::NonLinearExpression`] if the expression is not linear.
	pub fn linear_normal_form(&self) -> Result<LinearNormalForm<F>, Error> {
		if self.degree() > 1 {
			return Err(Error::NonLinearExpression);
		}
		self.evaluate()

	}

	// TODO impl from trait 
	fn evaluate(&self) -> Result<LinearNormalForm<F>, Error> {
		match self {
			Self::Const(val) => Ok(LinearNormalForm::new(*val)),
			Self::Var(index) => {
				let mut var_coeffs = vec![F::ZERO; self.n_vars()];
				var_coeffs[*index] = F::ONE;
				Ok(LinearNormalForm { constant: F::ZERO, var_coeffs})
			},
			Self::Add(left, right) => Ok(left.evaluate()? + right.evaluate()?),
			Self::Mul(left, right) => {
				left.evaluate()? * right.evaluate()?
			},
			Self::Pow(base, exp) => base.evaluate()?.pow(*exp as usize),
		}
	}

	   /// Evaluate to a dual‐number, detecting non‐linearity on the fly.
	pub fn evaluate_dual(&self, vars: &[Dual<F>]) -> Result<Dual<F>, Error> {
        match self {
            Self::Const(c)    => Ok(Dual::constant(*c, vars.len())),
            Self::Var(i)      => Ok(vars[*i].clone()),
            Self::Add(l, r)   => {
                let left = l.evaluate_dual(vars)?;
                let right = r.evaluate_dual(vars)?;
                Ok(left + right)
            }
            Self::Mul(l, r)   => {
                let left = l.evaluate_dual(vars)?;
                let right = r.evaluate_dual(vars)?;
                left.checked_mul(right)
            }
            Self::Pow(b, exp) => {
                let b = b.evaluate_dual(vars)?;
                b.pow_uint(*exp as usize)
            }
        }
    }

	 /// A convenience: build the dual inputs and extract a linear normal form
	 pub fn to_linear_via_dual(&self) -> Result<LinearNormalForm<F>, Error> {
        let n = self.n_vars();
        // xi ↦ Dual::variable(i,n)
        let inputs: Vec<Dual<F>> = (0..n)
            .map(|i| Dual::variable(i, n))
            .collect();

        let out = self.evaluate_dual(&inputs)?;
        Ok(LinearNormalForm {
            constant: out.real,
            var_coeffs: out.derivs,
        })
    }

	/// Returns a vector of booleans indicating which variables are used in the expression.
	///
	/// The vector is indexed by variable index, and the value at index `i` is `true` if and only
	/// if the variable is used in the expression.
	pub fn vars_usage(&self) -> Vec<bool> {
		let mut usage = vec![false; self.n_vars()];
		self.mark_vars_usage(&mut usage);
		usage
	}

	fn mark_vars_usage(&self, usage: &mut [bool]) {
		match self {
			Self::Const(_) => (),
			Self::Var(index) => usage[*index] = true,
			Self::Add(left, right) | Self::Mul(left, right) => {
				left.mark_vars_usage(usage);
				right.mark_vars_usage(usage);
			}
			Self::Pow(base, _) => base.mark_vars_usage(usage),
		}
	}



}


impl<F: TowerField> ArithExpr<F> {
	pub fn binary_tower_level(&self) -> usize {
		match self {
			Self::Const(value) => value.min_tower_level(),
			Self::Var(_) => 0,
			Self::Add(left, right) | Self::Mul(left, right) => {
				left.binary_tower_level().max(right.binary_tower_level())
			}
			Self::Pow(base, _) => base.binary_tower_level(),
		}
	}
}

impl<F> Default for ArithExpr<F>
where
	F: Field,
{
	fn default() -> Self {
		Self::zero()
	}
}

impl<F> Add for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		Self::Add(Box::new(self), Box::new(rhs))
	}
}

impl<F> AddAssign for ArithExpr<F>
where
	F: Field,
{
	fn add_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) + rhs;
	}
}

impl<F> Sub for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		Self::Add(Box::new(self), Box::new(rhs))
	}
}

impl<F> SubAssign for ArithExpr<F>
where
	F: Field,
{
	fn sub_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) - rhs;
	}
}

impl<F> Mul for ArithExpr<F>
where
	F: Field,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		Self::Mul(Box::new(self), Box::new(rhs))
	}
}

impl<F> MulAssign for ArithExpr<F>
where
	F: Field,
{
	fn mul_assign(&mut self, rhs: Self) {
		*self = std::mem::take(self) * rhs;
	}
}

impl<F: Field> Sum for ArithExpr<F> {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.reduce(|acc, item| acc + item).unwrap_or(Self::zero())
	}
}

impl<F: Field> Product for ArithExpr<F> {
	fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.reduce(|acc, item| acc * item).unwrap_or(Self::one())
	}
}

/// A normal form for a linear expression.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LinearNormalForm<F: Field> {
	/// The constant offset of the expression.
	pub constant: F,
	/// A vector mapping variable indices to their coefficients.
	pub var_coeffs: Vec<F>,
}

impl<F: Field> LinearNormalForm<F>{
	pub fn new(value: F) -> Self {
		Self {
			constant: value,
			var_coeffs: Vec::new(),
		}
	}

	pub fn constant(constant: F, n: usize) -> Self {
        Self { 
			constant,
			var_coeffs: vec![F::ZERO; n] 
		}
    }

	pub fn coeffs_constant(&self) -> bool {
		self.var_coeffs.iter().all(|&d| d == F::ZERO)
	}

	pub fn pow(self, exp: usize) -> Result<Self, Error> {
        match exp {
            0 => Ok(Self::constant(F::ONE, self.var_coeffs.len())),
            1 => Ok(self),
            _ => {
                // base must be constant to stay linear
                if self.coeffs_constant() {
                    Ok(Self {
                        constant:   self.constant.pow(exp as u64),
                        var_coeffs: vec![F::ZERO; self.var_coeffs.len()],
                    })
                } else {
                    Err(Error::NonLinearExpression)
                }
            }
        }
    }
}

// We can still use standard `+` for addition:
impl<F: Field> Add for LinearNormalForm<F> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
		let constant = self.constant + other.constant;
		// TODO speed it up 
        let var_coeffs = self.var_coeffs
            .into_iter()
            .zip(other.var_coeffs)
            .map(|(a,b)| a + b)
            .collect();
        Self {
            constant,
            var_coeffs,
        }
    }
}

   
impl<F: Field> Mul for LinearNormalForm<F> {
	type Output = Result<Self, Error>;
	fn mul(self, other: Self) -> Self::Output {
		let n = self.var_coeffs.len();
		let left_const  = self.coeffs_constant();
		let right_const = other.coeffs_constant();
	
		let constant = self.constant * other.constant;
		let var_coeffs = match (left_const, right_const) {
				(true, true)   => vec![F::ZERO; n],
				(true, false)  => other.var_coeffs.iter().map(|&d| self.constant * d).collect(),
				(false, true)  => self.var_coeffs.iter().map(|&d| d * other.constant).collect(),
				(false, false) => return Err(Error::NonLinearExpression),
			};
	
		Ok(Self { constant, var_coeffs })
	}
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_field::{BinaryField, BinaryField128b, BinaryField1b, BinaryField8b};

	use super::*;

	#[test]
	fn test_degree_with_pow() {
		let expr = ArithExpr::Const(BinaryField8b::new(6)).pow(7);
		assert_eq!(expr.degree(), 0);

		let expr: ArithExpr<BinaryField8b> = ArithExpr::Var(0).pow(7);
		assert_eq!(expr.degree(), 7);

		let expr: ArithExpr<BinaryField8b> = (ArithExpr::Var(0) * ArithExpr::Var(1)).pow(7);
		assert_eq!(expr.degree(), 14);
	}

	#[test]
	fn test_leading_term_with_degree() {
		let expr = ArithExpr::Var(0)
			* (ArithExpr::Var(1)
				* ArithExpr::Var(2)
				* ArithExpr::Const(BinaryField8b::MULTIPLICATIVE_GENERATOR)
				+ ArithExpr::Var(4))
			+ ArithExpr::Var(5).pow(3)
			+ ArithExpr::Const(BinaryField8b::ONE);

		let expected_expr = ArithExpr::Var(0)
			* ((ArithExpr::Var(1) * ArithExpr::Var(2))
				* ArithExpr::Const(BinaryField8b::MULTIPLICATIVE_GENERATOR))
			+ ArithExpr::Var(5).pow(3);

		assert_eq!(expr.leading_term_with_degree(), (3, expected_expr));
	}

	#[test]
	fn test_remap_vars_with_too_few_vars() {
		type F = BinaryField8b;
		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(1)).pow(3);
		assert_matches!(expr.remap_vars(&[5]), Err(Error::IncorrectArgumentLength { .. }));
	}

	#[test]
	fn test_remap_vars_works() {
		type F = BinaryField8b;
		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(1)).pow(3);
		let new_expr = expr.remap_vars(&[5, 3]);

		let expected = ((ArithExpr::Var(5) + ArithExpr::Const(F::ONE)) * ArithExpr::Var(3)).pow(3);
		assert_eq!(new_expr.unwrap(), expected);
	}

	#[test]
	fn test_optimize_identity_handling() {
		type F = BinaryField8b;
		let zero = ArithExpr::<F>::zero();
		let one = ArithExpr::<F>::one();

		assert_eq!((zero.clone() * ArithExpr::<F>::Var(0)).optimize(), zero);
		assert_eq!((ArithExpr::<F>::Var(0) * zero.clone()).optimize(), zero);

		assert_eq!((ArithExpr::<F>::Var(0) * one.clone()).optimize(), ArithExpr::Var(0));
		assert_eq!((one * ArithExpr::<F>::Var(0)).optimize(), ArithExpr::Var(0));

		assert_eq!((ArithExpr::<F>::Var(0) + zero.clone()).optimize(), ArithExpr::Var(0));
		assert_eq!((zero.clone() + ArithExpr::<F>::Var(0)).optimize(), ArithExpr::Var(0));

		assert_eq!((ArithExpr::<F>::Var(0) + ArithExpr::Var(0)).optimize(), zero);
	}

	#[test]
	fn test_const_subst_and_optimize() {
		// NB: this is FlushSumcheckComposition from the constraint_system
		type F = BinaryField8b;
		let expr = ArithExpr::Var(0) * ArithExpr::Var(1) + ArithExpr::one() - ArithExpr::Var(1);
		assert_eq!(expr.const_subst(1, F::ZERO).optimize().constant(), Some(F::ONE));
	}

	#[test]
	fn test_expression_upcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr = ((ArithExpr::Var(0) + ArithExpr::Const(F8::ONE))
			* ArithExpr::Const(F8::new(222)))
		.pow(3);

		let expected =
			((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Const(F::new(222))).pow(3);
		assert_eq!(expr.convert_field::<F>(), expected);
	}

	#[test]
	fn test_expression_downcast() {
		type F8 = BinaryField8b;
		type F = BinaryField128b;

		let expr =
			((ArithExpr::Var(0) + ArithExpr::Const(F::ONE)) * ArithExpr::Const(F::new(222))).pow(3);

		assert!(expr.try_convert_field::<BinaryField1b>().is_err());

		let expected = ((ArithExpr::Var(0) + ArithExpr::Const(F8::ONE))
			* ArithExpr::Const(F8::new(222)))
		.pow(3);
		assert_eq!(expr.try_convert_field::<BinaryField8b>().unwrap(), expected);
	}

	#[test]
	fn test_linear_normal_form() {
		type F = BinaryField128b;
		use ArithExpr::{Const, Var};

		let expr = Const(F::new(133))
			+ Const(F::new(42)) * Var(0)
			+ Var(2) + Const(F::new(11)) * Const(F::new(37)) * Var(3);
		let normal_form = expr.linear_normal_form().unwrap();
		assert_eq!(normal_form.constant, F::new(133));
		assert_eq!(
			normal_form.var_coeffs,
			vec![F::new(42), F::ZERO, F::ONE, F::new(11) * F::new(37)]
		);

		let expr = Const(F::ZERO);
		let normal_form = expr.linear_normal_form().unwrap();
        assert_eq!(normal_form.constant, F::ZERO);
        assert!(normal_form.var_coeffs.is_empty());

		let expr: ArithExpr<F> = Var(2);
		let normal_form = expr.linear_normal_form().unwrap();
        assert_eq!(normal_form.constant, F::ZERO);
        assert_eq!(normal_form.var_coeffs, vec![F::ZERO, F::ZERO, F::ONE]);

		let expr: ArithExpr<F> = Var(0) + Var(3);
		let normal_form = expr.linear_normal_form().unwrap();
		assert_eq!(normal_form.constant, F::ZERO);
		assert_eq!(normal_form.var_coeffs, vec![F::ONE, F::ZERO, F::ZERO, F::ONE]);

		 let expr: ArithExpr<F> = (Var(0) + Const(F::new(5))) * Const(F::new(3));
		 let normal_form =  expr
		 .linear_normal_form()
		 .unwrap();
	 	 assert_eq!(normal_form.constant, F::new(15));
	     assert_eq!(normal_form.var_coeffs, vec![F::new(3)]);

		 let expr: ArithExpr<F> = Const(F::new(7))
                    + Const(F::new(2)) * Var(1)
                    + Const(F::new(4)) * Var(0);
		let normal_form = expr.linear_normal_form()
            .unwrap();
        assert_eq!(normal_form.constant, F::new(7));
        assert_eq!(normal_form.var_coeffs, vec![F::new(4), F::new(2)]);

		let expr: ArithExpr<F> = Const(F::new(2)) * (Var(1) + Var(2));
        let normal_form = expr
            .linear_normal_form()
            .unwrap();
        assert_eq!(normal_form.constant, F::ZERO);
        assert_eq!(normal_form.var_coeffs, vec![F::ZERO, F::new(2), F::new(2)]);

		let expr: ArithExpr<F> = Var(0) * Var(1);
		let result_err = expr.linear_normal_form();
        assert!(matches!(result_err, Err(Error::NonLinearExpression)));

        let expr: ArithExpr<F> = (Var(0) + Const(F::ONE)).pow(2);
        let result_err = expr.linear_normal_form();
        assert!(matches!(result_err, Err(Error::NonLinearExpression)));
    }


}

