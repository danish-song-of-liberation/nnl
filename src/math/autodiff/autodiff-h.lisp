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

(defun + (&rest args)
  (reduce #'ll-add args))
  
(defun - (&rest args)
  (reduce #'intern-sub args))
  
(defun * (&rest args)
  (reduce #'ll-mul args))
  
(defun item (obj)
  (data obj))

(defun gradient (obj)
  (grad obj))  
  