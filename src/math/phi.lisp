(in-package :nnl.math)

#| Philipp Mainlender (1841 - 1876)
   - 
   Richtig und falsch sind Begriffe, 
   die im Zustand der Natur keine Bedeutung haben |#

(declaim (inline relu))
(declaim (inline leaky-relu))
(declaim (inline sigmoid))
(declaim (inline tanh))
(declaim (inline liinear))

(defun relu (x &key (derivative nil))
  "Rectified Linear Unit (ReLU) function.	
  
   returns x if x > 0, otherwise returns 0

   example 1: (relu 2.0) -> (if (plusp 2.0) 2.0 0.0) -> 2.0
   example 2: (relu -3.0) -> (if (plusp -3.0) -3.0 0.0) -> 0.0

   if derivative non-nil, returns derivative):

   if x > 0 returns 1 otherwise returns 0

   example 1: (relu 2.0 :derivative t) -> (if (plusp 2.0) 1.0 0.0) -> 1.0
   example 2: (relu -3.0 :derivative t) -> (if (plusp 3.0) 1.0 0.0) -> 0.0"

  (if derivative
    (if (plusp x) 1.0 0.0)
    (if (plusp x) x 0.0))) ; (max x 0.0)

(defun leaky-relu (x &key (derivative nil) (shift nnl.system::*leakyrelu-default-shift*))
  "Leaky Rectified Linear Unit (LReLU) function.

   returns x if x > 0, otherwise returns shift(default 0.01) * x

   example 1: (leaky-relu 4.0) -> (max 4.0 0.04) -> 4.0
   example 2: (leaky-relu -3.0) -> (max -3.0 -0.03) -> -0.03

   derivative:

   if x > 0 then return 1 else shift

   example 1: (leaky-relu 4.0 :derivative t) -> (if (plusp 4.0) 1.0 0.01) -> 1.0
   example 2: (leaky-relu -3.0 :derivative t) -> (if (plusp -3.0) 1.0 0.01) -> 0.01
   
   to change the shift write (setf nnl.system::*leakyrelu-default-shift* +your-value+)"

  (if derivative
    (if (plusp x) 1.0 shift)
    (if (minusp x) (cl:* shift x) x))) ; (max x (* shift x))

(defun sigmoid (x &key (derivative nil))
  "Sigmoid function and its derivative.

   check it: https://www.desmos.com/calculator/3tspufomdy?lang=eng

   sigmoid function squashes a real-valued input to a range between 0 and 1.
   formula: sigmoid(x) = 1/(1 + exp(-x))

   example 1: (sigmoid 0.0) -> 0.5
   example 2: (sigmoid 1.0) ~> 0.731
   example 3: (sigmoid -1.0) ~> 0.268

   derivative formula: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

   example 1: (sigmoid 0.0 :derivative t) -> 0.25
   example 2: (sigmoid 1.5 :derivative t) ~> 0.149
   example 3: (sigmoid -2.0 :derivative t) ~> 0.104" 

  (if derivative
    (let ((sig (sigmoid x)))
      (cl:* sig (cl:- 1.0 sig)))
	  
    (cl:/ 1.0 (cl:+ 1.0 (cl:exp (cl:- x))))))

(defun tanh (x &key (derivative nil))
  "Hyperbolic tangent function (tanh).

   check it: https://www.desmos.com/calculator/eai4bialus?lang=eng

   tanh function squashes a real-valued input to a range between -1 and 1
   
   tanh formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

   example 1: (tanh 1.0) -> 0.761
   example 2: (tanh 2.5) -> 0.986
   example 3: (tanh -1.5) -> -0.905

   derivative formula: tanh'(x) = tanh(x) * (1 - tanh(x))

   example 1: (tanh 1.0 :derivative t) -> 0.419
   example 2: (tanh 2.5 :derivative t) -> 0.026
   example 3: (tanh -1.5 :derivative t) -> 0.180"

  (if derivative
    (cl:- 1.0 (cl:* (tanh x) (tanh x)))
    (cl:tanh x)))

(defun linear (x &key (derivative nil))
  "Linear function
   
   just returns x
  
   example 1: (linear 0.3) -> 0.3
   example 2: (linear 0.4) -> 0.4
   
   derivative: just 1
   
   example 1: (linear 0.5 :derivative t) -> 1.0
   example 2: (linear 0.8 :derivative t) -> 1.0"
   
  (if derivative
    1.0
    x))
