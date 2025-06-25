(in-package :nnl.math.autodiff)

(defmacro make-tensor (&body args)
  `(make-instance 'tensor ,@args))

(defmethod repr ((self tensor))
  (format nil "Tensor ~a:~%~%
			   Value (data): ~f~%
			   Grad-required: ~a~%
			   Parents: ~a~%
			   Backward: ~a~%
			   Grad (may be incorrect if backpropagation has not yer been carried out): ~a~%"
			   
	self (data self) (requires-grad self) (parents self) (backward self) (grad self)))

(defun repr! (tensor)
  (format t "~%~a~%" (repr tensor)))

(defmethod ll-add ((self tensor) other)
  (intern-add self other))

(defmethod ll-add (other (self tensor))
  (intern-add self other))
  
(defmethod ll-mul ((self tensor) other)
  (intern-mul self other))
 
(defmethod ll-mul (other (self tensor))
  (intern-mul self other))
  
(defmethod ll-matmul ((self tensor) other)
  (intern-matmul self other))
  
(defmethod ll-matmul (other (self tensor))
  (intern-matmul other self))

(defun + (&rest args)
  (reduce #'ll-add args))
  
(defun - (&rest args)
  (reduce #'intern-sub args))
  
(defun * (&rest args)
  (reduce #'ll-mul args))  
  
(defun matmul (&rest args)
  (reduce #'ll-matmul args))
  
(defmethod axpy ((self tensor) other &key (alpha 1.0))
  (intern-axpy self other :alpha alpha))

(defmethod axpy (other (self tensor) &key (alpha 1.0))
  (intern-axpy other self :alpha alpha))
  
(defmethod ll-hstack ((self tensor) other)
  (intern-hstack other self))  
  
(defmethod ll-hstack (other (self tensor))
  (intern-hstack other self))  
  
(defun hstack (&rest args)
  (reduce #'ll-hstack args))  
  
(defmethod ll-vstack ((self tensor) other)
  (intern-vstack other self))  
  
(defmethod ll-vstack (other (self tensor))
  (intern-vstack other self))  
  
(defun vstack (&rest args)
  (reduce #'ll-vstack args))    
  
(defun item (obj)
  (data obj))

(defun gradient (obj)
  (grad obj))  

(defmacro step (obj &key (lr 0.1))
  `(magicl:.- (data ,obj) (magicl:scale (grad ,obj) ,lr)))
  
(defmacro step! (obj &key (lr 0.1))
  `(setf (data ,obj) (step ,obj :lr ,lr)))
  