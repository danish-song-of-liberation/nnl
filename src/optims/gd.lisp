(in-package :nnl.optims)

(defclass intern-gd ()
  ((%parameters 
	:initform '()
	:initarg :parameters
	:accessor parameters
	:type list)
   (%learning-speed
    :initform 0.1
	:initarg :lr
	:accessor lr
	:type real)))
	
(defmethod backpropagation ((optimizer intern-gd) nn (examples nnl.math.autodiff:tensor) (tags nnl.math.autodiff:tensor) &key (epochs 1) (plugins '()) (loss #'nnl.math.autodiff:mse))
  (let* ((forward-pass (funcall (nnl.nn:forward-method nn) nn examples))
		 (forward-loss (funcall loss forward-pass tags)))

	(nnl.math.autodiff:backprop forward-loss)
	
    (dolist (model (parameters optimizer))
	  (dolist (parameter model)
	    (dolist (plugin plugins)
		  (funcall plugin parameter))
		  
	    (nnl.math.autodiff:step! parameter :lr (lr optimizer))))
		
	(nnl.math.autodiff:zero-grad! forward-loss)
	
    (when (and (/= epochs 1) (not (< epochs 1))) (backpropagation optimizer nn examples tags :epochs (- epochs 1) :plugins plugins :loss loss))))
    
	