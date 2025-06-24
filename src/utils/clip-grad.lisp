(in-package :nnl.utils)

(defmethod intern-threashold-gradient-clipping ((self nnl.math.autodiff:tensor) &key (threashold nnl.system::*clipgrad-l1-default-threashold*))
  (magicl:map! #'(lambda (x) (cond ((> x threashold) threashold) ((< x (- threashold)) (- threashold)) (t x))) (nnl.math.autodiff::grad self)))
  
(defmethod clip-grad-once ((self nnl.math.autodiff:tensor) &key (method :l1))
  (case method
    (:l1 (intern-threashold-gradient-clipping self))))

(defun clip-grad (parameters &key (method :l1))
  "todo optimize"

  (if (listp parameters)
    (mapcar #'(lambda (x) (clip-grad x :method method)) parameters)
	(clip-grad-once parameters :method method)))
  