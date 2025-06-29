```
 .--..--..--..--..--..--..--. 
/ .. \.. \.. \.. \.. \.. \.. \
\ \/\ `'\ `'\ `'\ `'\ `'\ \/ /
 \/ /`--'`--'`--'`--'`--'\/ / 
 / /\                    / /\ 
/ /\ \              _   / /\ \
\ \/ /  _ __  _ __ | |  \ \/ /
 \/ /  | '_ \| '_ \| |   \/ / 
 / /\  | | | | | | | |   / /\ 
/ /\ \ |_| |_|_| |_|_|  / /\ \
\ \/ /                  \ \/ /
 \/ /                    \/ / 
 / /\.--..--..--..--..--./ /\ 
/ /\ \.. \.. \.. \.. \.. \/\ \
\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
 `--'`--'`--'`--'`--'`--'`--' 
 ```

# nnl 0.1.0 (W.I.P.)
Сommon lisp neural networks mvp framework

**About the Author:** This framework is being developed by a 14-year-old as a personal *solo* project. It's currently 13 days (actually 12 because 1 of them I was resting) in the making! *All code, all bugs, all mine!*

I write the framework mainly for myself because writing on torch or tensorflow develops into eternal procrastination and unwillingness to write code on it.

## Note

The code is formatted in Notepad++, so the indentation may look like a cat did it on the keyboard.

## Note

This is an alpha version! API may change.

## Note on Documentation Completeness:
This documentation is currently a work in progress. 
As a solo developer, I'm actively expanding and refining the content. 
Full documentation covering all features (including advanced autodiff capabilities and model architectures) will be released in upcoming versions. 

Your patience is appreciated!

## Customizing the Calculus System

By default, `nnl` uses `single-float` for all numerical computations.  However, you can easily change the calculus system to use a different floating-point type, such as `double-float`, to increase precision.

To change the calculus system, use the `nnl.system:change-calculus-system!` function.  This function accepts a single argument: a symbol representing the desired floating-point type.

**Example:**

To switch to using `double-float`:

```lisp
(nnl.system:change-calculus-system! 'double-float)
```

## Working with NNL Tensors

NNL provides a flexible and intuitive way to work with tensors, similar to NumPy and torch. Here are some examples:

`(nnl.math:item tensor)` - Returns the underlying data of the tensor.

**Example:**
```lisp
(nnl.math:item (nnl.math:make-tensor #(1 2 3))) ; Returns magicl vector #(1 2 3)
```

### Creating Tensors

NNL provides functions to create tensors with specific shapes and values.

*   `(nnl.math:zeros '(3 2))` - Creates a tensor filled with zeros with shape (3, 2).
*   `(nnl.math:ones '(3 2))` - Creates a tensor filled with ones with shape (3, 2).
*   `(nnl.math:arange start end &optional step &key requires-grad)` - Creates a tensor with a sequence of numbers. `start` is the starting value, `end` is the ending value (exclusive), and `step` is the increment.

**Example:**
```lisp
(nnl.math:zeros '(3 2) :requires-grad nil) ; Creates a 3x2 tensor of zeros
(nnl.math:arange 0 10 2 :requires-grad t) ; Creates a tensor: #(0 2 4 6 8) with requires-grad set to T
```

### Basic Tensor Operations

NNL allows you to perform basic arithmetic operations on tensors using standard operators. Just replace the package with `nnl.math`.

**Example:**
```lisp
(nnl.math:+ (nnl.math:ones '(2 2)) (nnl.math:ones '(2 2))) ; Adds two 2x2 tensors filled with ones
```

### Matrix Operations

NNL provides functions for matrix multiplication.

*   `(nnl.math:matmul matrix1 matrix2)` - Performs matrix multiplication.
*   `(nnl.math:matvec matrix vector)` - Multiplies a matrix by a vector.

**Example:**
```lisp
(nnl.math:matmul (nnl.math:ones '(2 2)) (nnl.math:ones '(2 2))) ; Matrix multiplication
```

### Creating Tensors from Arrays

You can create tensors from Common Lisp arrays using `nnl.math:make-tensor`.

**Example:**
```lisp
(nnl.math:make-tensor #(1 2 3)) ; Creates a 1D tensor from a vector
(nnl.math:make-tensor #2A((1 2) (3 4))) ; Creates a 2D tensor from a 2x2 array
(nnl.math:make-tensor #3A(((1 2) (3 4)) ((5 6) (7 8)))) ; Creates a 3D tensor
```

### Example of Backpropagation

This example demonstrates how to use `requires-grad` and perform backpropagation in NNL.

**Code:**
```lisp
(ql:quickload :nnl)

(defun forward (x y)
  (nnl.math:matmul x y))

(let ((a (nnl.math:make-tensor #2A((1 2) (3 4)) :requires-grad t))
      (b (nnl.math:make-tensor #2A((5 6) (7 8)) :requires-grad t))
      (targ (nnl.math:make-tensor #2A((9 8) (7 6))))) 

  (let ((loss (nnl.math.autodiff:mse (forward a b) targ))) ; (a - b)^2
    (nnl.math:backprop loss)

    (format t "a Gradient: ~a~%" (nnl.math:grad a)) ; #<matrix/single-float (2x2): 268, 364, 888, 1208>
    (format t "b Gradient: ~a~%" (nnl.math:grad b)))) ; #<matrix/single-float (2x2): 236, 292, 328, 408>
```

**Explanation:**

1.  We first load the `nnl` library using `ql:quickload`.
2.  We define a simple `forward` function that performs matrix multiplication using `nnl.math:matmul`.
3.  We create three tensors `a`, `b`, and `targ` using `nnl.math:make-tensor`. We set `:requires-grad t` for `a` and `b` to indicate that we want to compute gradients for these tensors.
4.  We calculate the mean squared error (MSE) loss between the result of the `forward` function and the `targ` tensor using `nnl.math.autodiff:mse`.
5.  We call `nnl.math:backprop` on the `loss` tensor to perform backpropagation. This computes the gradients of the `loss` with respect to all tensors that have `requires-grad` set to `T`.
6.  Finally, we print the gradients of `a` and `b` using `nnl.math:grad`.

**Key Concepts:**

*   **`:requires-grad t`:** This tells NNL to track operations on a tensor so that gradients can be computed.
*   **`nnl.math:backprop`:** This performs the backpropagation algorithm, computing gradients for all tensors that require them.
*   **`nnl.math:grad`:** This returns the gradient of a tensor.


### Zeroing Gradients

After performing backpropagation, it's essential to zero the gradients of the tensors before the next iteration. NNL provides two functions for this purpose:

*   **`nnl.math:zero-grad! tensor`**: Zeros the gradients of *all* tensors that were involved in the computation of `tensor` during the forward pass.

*   **`nnl.math:zero-grad-once! tensor`**: Zeros the gradient of *only* the specified `tensor`.

**Example:**
```lisp
(nnl.math:backprop loss)
(nnl.math:zero-grad! loss) ; Zeros the gradients of all tensors involved in calculating loss
```

**Example (Zeroing a specific tensor):**
```lisp
(let ((foo (nnl.math:make-tensor #(1 2 3) :requires-grad t)))
  ; ... perform some operations with foo ...
  (nnl.math:backprop some-loss)
  (nnl.math:zero-grad-once! foo)) ; Zeros only the gradient of foo
```

**Explanation:**

*   `zero-grad!` is typically used after each training iteration to reset the gradients before calculating them again for the next batch of data.
*   `zero-grad-once!` can be useful if you want to exclude certain tensors from the gradient update, or if you want to manually manipulate the gradients of specific tensors.

**Important Note:**

Do not forget to reset the gradients (otherwise you'll get a stack dump after 5 epochs)




**Addition:**

The framework contains methods for numerical gradients, but they will be discussed in the full documentation that is under development.
Here is a brief overview: numerical gradients are more compatible with magicl than with nnl (although they are partially compatible with both). 

**Here is an example of code in mse:**
```lisp
(ql:quickload :nnl)

(defun nummse (x)
  (nnl.math:instant-mse x (nnl.magicl:coerce-to-tensor #(2 2))))

(let ((a (nnl.magicl:coerce-to-tensor #(1 1)))
      (b (nnl.math:make-tensor #(1 1) :requires-grad t)))

  (nnl.math:backprop (nnl.math.autodiff:mse b (nnl.math:make-tensor #(2 2))))

  (format t "~%Autodiff grad: ~a~%~%" (nnl.math:grad b))

  (format t "Numerical grad: ~a~%~%" (nnl.magicl:numerical #'nummse a :precision 1))) ; precision 1 - df/dx [i] = lim (h -> 0.0) (f(x + he_i) - f(x)) / h
```

### Accessing Tensor Elements

NNL provides several functions for accessing elements, subvectors, and submatrices within tensors. These functions support backpropagation and gradient computation when used with tensors that have :requires-grad t.

**Basic Element Access**

```lisp (nnl.math:tref tensor &rest indices)``` - Returns a scalar tensor containing the element at the specified indices.

**Example:**

```lisp
(let ((a (nnl.math:make-tensor #2A((1 2 3) (4 5 6)))))
  (print (nnl.math:item (nnl.math:tref a 1 2)))) ;; 6.0
```

**Subvector Access:**

```lisp (nnl.math:trefv tensor &rest indices)``` - Returns a vector tensor containing the subvector at the specified index.

**Example:**

```lisp
(let ((a (nnl.math:make-tensor #2A((1 2 3) (4 5 6)))))
  (print (nnl.math:item (nnl.math:trefv a 1)))) ;; #<magicl:vector/single-float (3): 4 5 6>
```

**Submatrix Access:**

```lisp (nnl.math:trefm tensor &rest indices)``` - Returns a matrix tensor containing the submatrix at the specified index (for tensors of rank 3 or higher).

**Example:**

```lisp
(let ((a (nnl.math:zeros '(3 3 3))))
  (print (nnl.math:item (nnl.math:trefm a 1)))) ;;#<magicl:matrix/single-float (3): 0 0 0 \n 0 0 0 \n 0 0 0>
```

**Backpropagation Support**

All access functions support backpropagation. When you access part of a tensor and perform operations on it, gradients will be properly propagated back to the original tensor.

**Example with Backpropagation:**

```lisp
(let* ((a (nnl.math:zeros '(3 3) :requires-grad t))
       (b (nnl.math:trefv a 0))
       (c (nnl.math:* b 3)))

  (print (nnl.math:item c)) ;; #<magicl:vector/single-float (3): 0 0 0>

  (nnl.math:backprop c)

  (terpri)

  (print (nnl.math:grad a))) ;; #<magicl:matrix/single-float (3x3): 3 3 3 \n 0 0 0 \n 0 0 0>
```

### Models: Current Status and Future Directions

NNL currently includes implementations of several common neural network models (e.g., `fc`, `sequential`). However, these models have a low-level interface and are not yet intended for general use. They are actively being developed and refined.

**Current Interface:**

The current interface for working with models is low-level and requires a deep understanding of the underlying tensor operations.

**Future Interface: A DSL-Oriented Approach:**

The future direction for NNL models is to provide a high-level, DSL (Domain Specific Language) oriented interface for defining neural network architectures. The goal is to make it easy to create and experiment with different models without having to write low-level code.

**High-Level Interface (nnl.hli) (Experimental)**

**Current Status:**

The high-level interface for model creation is partially implemented but still considered experimental. While you can already define models using the DSL-like syntax, i recommend against using it in production until the following issues are resolved:

1. Loss calculation requires manual tensor operations (will be simplified)

2. The API is still subject to change

Example of Experimental Interface (XOR):

```lisp
(ql:quickload :nnl)

(setf *random-state* (make-random-state t))

(let* ((a (nnl.hli:sequential
            (nnl.hli:fc 2 -> 2)
            (nnl.hli:fc 2 -> 2)
            (nnl.nn:tanh)
            (nnl.hli:fc 2 -> 1)
            (nnl.nn:sigmoid)))

        (input (nnl.math:make-tensor #2A((0 0) (1 0) (0 1) (1 1))))
        (target (nnl.math:make-tensor #2A((0) (1) (1) (0))))

        (epochs 1000)
        (params (nnl.nn:get-parameters a))

        (optim (nnl.optims:make-optim 'nnl.optims:momentum :lr 0.1 :parameters params)))

  (dotimes (i epochs)
    (let* ((forward-pass (nnl.nn:forward a input))
           (loss (nnl.math.autodiff:mse forward-pass target)))

      (nnl.math:backprop loss)

      (nnl.optims:step optim)
      (nnl.optims:zero-grad optim)))

  (print (magicl:map #'nnl.utils:binary-threashold (nnl.math:item (nnl.nn:forward a input))))) ; #(0 1 1 0)
```

**ABOUT MLP:**

The definition looks something like this:

```lisp
(nnl.hli:mlp '(2 -> 2 -> nnl.nn::intern-tanh -> 1 -> nnl.nn::intern-sigmoid) ... )
```

where ... is keys (like :init :xavier/uniform)

which is similar

```lisp
(nnl.hli:sequential
  (nnl.hli:fc 2 -> 2)
  (nnl.hli:fc 2 -> 2)
  (nnl.nn:tanh)
  (nnl.hli:fc 2 -> 1)
  (nnl.nn:sigmoid))
```


**TODO:**

Vanilla RNN, GRU, LSTM, BiRNN, BiGRU, BiLSTM, potentially transformers (I'm doing rnn now) <br>
optimize Autodiff<br>
make documentation<br>
make a dropout, dropconnect<br>
leave the docstring in most functions and seriously optimize them<br>
the ability to save and load neural networks<br>
add the ability to download datasets from the internet<br>
add pad-sequence, embeddings, one-hot<br>
add swish, mish, hardtanh, h-swish, relu6, linear, and the ability to add your own activation functions<br>
add cross-entropy, binary-cross-entropy<br>
add tests<br>

<br><br><br><br><br><br><br><br><br><br>








<h3 style="color: #5a1a8c; margin-top: 0;">Philipp Mainländer (1841–1876)</h3>

<blockquote style="font-style: italic; border-left: 3px solid #b57edc; padding-left: 15px; margin-left: 0;">
<p>Jede Handlung des Menschen, die höchste wie die niedrigste, ist egoistisch; denn sie entspringt einer bestimmten Individualität, einem bestimmten Ich, mit einem ausreichenden Motiv und kann in keiner Weise ausgelassen werden. Auf den Grund der Verschiedenheit der Charaktere einzugehen, ist hier nicht der richtige Ort; wir müssen es einfach als Tatsache akzeptieren.</p>

<p>Nun ist es für den barmherzigen Mann genauso unmöglich, seinen Nächsten hungern zu lassen, wie es für den hartherzigen Mann ist, den Armen zu helfen. Jeder der beiden handelt nach seinem Charakter, seiner Natur, seinem Ego, seinem Glück, folglich egoistisch; denn wenn der Barmherzige die Tränen anderer nicht trocknen würde, wäre er glücklich? Und wenn der Hartherzige das Leiden anderer linderte, wäre er zufrieden?</p>
</blockquote>

<div style="background-color: #e9d8fd; padding: 15px; border-radius: 5px; margin-top: 15px;">
<h4 style="color: #4b1a7a; margin-top: 0;">English Translation:</h4>

<blockquote style="font-style: normal; border-left: 3px solid #9f6ec1; padding-left: 15px;">
<p>Every action of man, the highest as well as the lowest, is egoistic; for it flows from a certain individuality, a certain I, with a sufficient motive, and can in no way be omitted.</p>

<p>To go into the reason of the difference of characters is not the place here; we have simply to accept it as a fact. Now it is just as impossible for the merciful man to let his neighbor starve as it is for the hard-hearted man to help the poor.</p>

<p>Each of the two acts according to his character, his nature, his ego, his happiness, consequently egoistically; for if the merciful one did not dry the tears of others, would he be happy? And if the hard-hearted one relieved the suffering of others, would he be satisfied?</p>
</blockquote>
</div>

</div>