(in-package :nnl.nn)

(defclass intern-fc ()
  ((%weights 
    :initform nil 
	:accessor weights)
   (%bias 
    :initform nil 
	:accessor bias)
   (%use-bias 
    :initform t 
	:initarg :use-bias 
	:accessor use-bias)
   (%input-shapes 
    :initform 0 
	:initarg :i 
	:reader input-shapes)
   (%output-shapes 
    :initform 0 
	:initarg :o 
	:reader output-shapes)
   (%initialize-method
    :initform :xavier/normal
	:initarg :init
	:reader initialize-method)
   (%initialize-from
    :initform -1.0
	:initarg :from
	:reader init-from)
   (%initialize-to
    :initform 1.0
	:initarg :to
	:reader init-to))
   (:documentation "todo"))
   
(defmethod initialize-instance :after ((nn intern-fc) &key)
  (let ((input-shapes (input-shapes nn))
		(output-shapes (output-shapes nn))
		(use-bias-p (use-bias nn)))
		
	(assert (not (<= input-shapes 0)))
	(assert (not (<= output-shapes 0)))
	
	(case (initialize-method nn)
	  (:random 
	    (setf (weights nn) (nnl.init:random-initialize (list input-shapes output-shapes) :from (init-from nn) :to (init-to nn) :requires-grad t))
		
		(when use-bias-p 
		  (setf (bias nn) (nnl.init:random-initialize (list output-shapes) :from (/ (init-from nn) 10) :to (/ (init-to nn) 10) :requires-grad t))))
		  
	  (:xavier/normal
	    (setf (weights nn) (nnl.init:xavier-normal-initialize (list input-shapes output-shapes) (+ input-shapes output-shapes) :requires-grad t))
		
		(when use-bias-p
		  (setf (bias nn) (nnl.math:zeros (list output-shapes) :requires-grad t))))
		  
	  (:xavier/uniform
	    (setf (weights nn) (nnl.init:xavier-uniform-initialize (list input-shapes output-shapes) (+ input-shapes output-shapes) :requires-grad t))
		
		(when use-bias-p
		  (setf (bias nn) (nnl.math:scale (nnl.init:xavier-uniform-initialize (list output-shapes) (+ input-shapes output-shapes) :requires-grad t) 0.1)))))))
		  
(defmethod forward ((nn intern-fc) input-data &key padding-mask) ;todo padding mask (tbh its useless in fc)  
  (let ((result (nnl.math:matmul input-data (weights nn)))) ;; weights already transposed in initialization
    (if (use-bias nn)
	  (nnl.math:with-broadcasting #'+ result (bias nn))
	  result)))

(defmethod get-parameters ((nn intern-fc))
  (if (use-bias nn)
    (list (weights nn) (bias nn))
	(list (weights nn))))
	
(defmethod set-bias ((nn intern-fc) state)
  (setf (use-bias nn) state))	
   
(defmacro fc (&body keys)
  `(make-instance 'intern-fc ,@keys))   
   