(in-package :nnl.math.autodiff)

(defmethod tref ((self tensor) &rest pos) 
  (let ((out (make-instance 'tensor 
		      :data (apply #'magicl:tref (data self) pos)
			  :requires-grad (requires-grad self)
			  :parents (list self))))
	
	(setf (backward out) #'(lambda () (derivative-tref out self pos))) 
	
	out))
		
(defun ll-trew (tensor &rest pos)
  "Tensor ref With subtensors support"
  
 )
		