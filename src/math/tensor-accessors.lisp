(in-package :nnl.math.autodiff)

(defmethod tref ((self tensor) &rest pos) 
  (let ((out (make-instance 'tensor 
		      :data (apply #'magicl:tref (data self) pos)
			  :requires-grad (requires-grad self)
			  :parents (list self))))
	
	(setf (backward out) #'(lambda () (derivative-tref out self pos))) 
	
	out))
		
(defmethod trefv ((self tensor) &rest pos)
  (let ((out (make-instance 'tensor 
		      :data (apply #'nnl.magicl:trefv (data self) pos)
			  :requires-grad (requires-grad self)
			  :parents (list self))))
	
	(setf (backward out) #'(lambda () (derivative-trefv out self pos))) 
	
	out))

(defmethod trefm ((self tensor) &rest pos)
  (let ((out (make-instance 'tensor 
		      :data (apply #'nnl.magicl:trefm (data self) pos)
			  :requires-grad (requires-grad self)
			  :parents (list self))))
	
	(setf (backward out) #'(lambda () (derivative-trefm out self pos))) 
	
	out))
	