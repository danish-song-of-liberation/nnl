(in-package :nnl.math.autodiff)

(defclass tensor ()
  ((%data 
	:initform nil
	:accessor data
	:initarg :data
	:documentation "Tensor that stores computational data.
					The type is not specified due to the separation of data types.
					type = magicl:tensor (or matrix/vector)/single-float (or double-float)")
   (%grad 
	:initform nil
	:accessor grad
	:documentation "Gradient of the tensor
					The type is not specified due to the separation of data types.
					type = magicl:tensor (or matrix/vector)/single-float (or double-float)")
   (%parents 
    :initform '() 	
	:accessor parents
	:initarg :parents
	:type list
	:documentation "Tensor parents for backpropagation")
   (%requires-grad 
    :initform nil
	:reader autograd
	:initarg :requires-grad
	:type boolean
	:documentation "in the case of nil, gradients will not be calculated. in the case of t, there will be")
   (%backward 
    :initform #'(lambda ())
	:accessor backward
	:type function
	:documentation "function for backpropagation")))
