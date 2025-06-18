(in-package :nnl.nn.fnn)

(defclass intern-fc ()
  ((%input-shapes
	:initform 0
	:initarg :i
	:reader input-shapes
	:type integer)
   (%output-shapes
	:initform 0
	:initarg :o
	:reader output-shapes
	:type integer)
   (%weights
	:initform nil
	:accessor weights
	:type (or null nnl.math.autodiff:tensor))
   (%bias
	:initform nil
	:accessor bias
	:type (or null nnl.math.autodiff:tensor))
   (%use-bias
    :initform t
	:initarg :bias
	:reader use-bias
	:type boolean)
   (%initialize-method
    :initform :xavier/normal
	:initarg :init
	:reader initialize-method
	:type symbol)
   (%init-from
    :initform -1.0
	:initarg :from
	:reader init-from
	:type real)
   (%init-to
    :initform 1.0
	:initarg :to
	:reader init-to
	:type real)
   (%forward-method
    :initform nil
	:accessor forward-method
	:type (or null function))))

(defmethod forward ((nn intern-fc) input-data)
  (let ((result (nnl.math:matmul (weights nn) input-data)))
    (if (use-bias nn)
	  (nnl.math:.+ result (bias nn))
	  result)))
	
(defmethod initialize-instance :after ((nn intern-fc) &key)
  (let ((input-shapes (input-shapes nn))
		(output-shapes (output-shapes nn)))
		
    (assert (/= 0 input-shapes))
    (assert (/= 0 output-shapes))

    (setf (forward-method nn) #'forward)
	
	(case (initialize-method nn)
	  (:random
	    (setf (weights nn) (nnl.init:random-initialize (list input-shapes output-shapes) :from (init-from nn) :to (init-to nn) :requires-grad t))
		
		(when (use-bias nn)
		  (setf (bias nn) (nnl.init:random-initialize (list output-shapes input-shapes) :from (init-from nn) :to (init-to nn) :requires-grad t))))
		  
	  (:xavier/normal
	    (setf (weights nn) (nnl.init:xavier-normal-initialize (list input-shapes output-shapes) (+ input-shapes output-shapes) :requires-grad t))
		
		(when (use-bias nn)
		  (setf (bias nn) (nnl.init:xavier-normal-initialize (list output-shapes input-shapes) (+ input-shapes output-shapes) :requires-grad t))))
		  
	  (:xavier/uniform
	    (setf (weights nn) (nnl.init:xavier-uniform-initialize (list input-shapes output-shapes) (+ input-shapes output-shapes) :requires-grad t))
		
		(when (use-bias nn)
		  (setf (bias nn) (nnl.init:xavier-uniform-initialize (list output-shapes input-shapes) (+ input-shapes output-shapes) :requires-grad t)))))))
		  
(defmethod get-parameters ((nn intern-fc))
  (list (list (weights nn) (bias nn))))		  
		  