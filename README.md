# nnl 0.1.0 (W.I.P.)
Сommon lisp neural networks mvp framework

## Note

Please be aware: this code was formatted using Notepad++.  
Due to the way it handles indentation,
the indentation might look a bit unusual.  
Please don't be alarmed!**.**

## Note

This is an alpha version! API may change.

## Customizing the Calculus System

By default, `nnl` uses `single-float` for all numerical computations.  However, you can easily change the calculus system to use a different floating-point type, such as `double-float`, to increase precision.

To change the calculus system, use the `nnl.system:change-calculus-system!` function.  This function accepts a single argument: a symbol representing the desired floating-point type.

**Example:**

To switch to using `double-float`:

```lisp
(nnl.system:change-calculus-system! 'double-float)
