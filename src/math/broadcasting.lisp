(in-package :nnl.math)

(defmethod with-broadcasting ((operand function) (obj-1 nnl.math.autodiff:tensor) (obj-2 nnl.math.autodiff:tensor))
  (multiple-value-bind (reduce shapes-obj1 form-obj1 shapes-obj2 form-obj2) (nnl.magicl:with-broadcast operand (nnl.math.autodiff::data obj-1) (nnl.math.autodiff::data obj-2))
    (let ((out (make-instance 'nnl.math.autodiff:tensor :data reduce
			            							    :requires-grad (or (nnl.math.autodiff::requires-grad obj-1) (nnl.math.autodiff::requires-grad obj-2))
								            		    :parents (list obj-1 obj-2))))
	
	  (setf (nnl.math.autodiff::backward out) #'(lambda () (nnl.math.autodiff::derivative-broadcasting operand out obj-1 shapes-obj1 form-obj1 obj-2 shapes-obj2 form-obj2)))
	
	 out)))
										 
										 
  