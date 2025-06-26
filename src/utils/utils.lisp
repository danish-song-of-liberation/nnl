(defpackage :nnl.utils
  (:use :cl)
  (:export :clip-grad :binary-threashold :average))
  
(in-package :nnl.utils)

(defun binary-threashold (x)
  (if (> x 0.5)
    1
    0))	
	
(defmethod average ((tensor nnl.math.autodiff:tensor) &aux (size 0))
  (magicl:map #'(lambda (x) (incf size)) (nnl.math.autodiff::data tensor)) ;WTF
  (/ (nnl.magicl:sum (nnl.math.autodiff::data tensor)) size))
	