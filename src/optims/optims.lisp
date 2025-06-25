(defpackage :nnl.optims
  (:use :cl)
  (:shadow step)
  (:export :step :zero-grad :make-optim :gd :momentum))

(in-package :nnl.optims)

(defmacro make-optim (name &body keys)
  `(make-instance ,name ,@keys))
  
(defun zero-parameters (parameters)
  "todo optimize"
  
  (if (listp parameters)
    (mapcar #'zero-parameters parameters)
    (nnl.math.autodiff:zero-grad-once! parameters)))	  
	