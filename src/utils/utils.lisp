(defpackage :nnl.utils
  (:use :cl)
  (:export :clip-grad :binary-threashold))
  
(in-package :nnl.utils)

(defun binary-threashold (x)
  (if (> x 0.5)
    1
    0))	
  