(defpackage :nnl.system
  (:use :cl)
  (:export :change-calculus-system! ~=))

(in-package :nnl.system)

(defparameter *calculus-system* 'single-float
  "Global variable that determines the floating-point 
   number system used for calculations within a specific 
   mathematical functionality")
   
(setf magicl::*default-tensor-type* 'single-float)   
   
(defparameter *leakyrelu-default-shift* 0.01)
(defparameter *approximately-equal-default-tolerance* 0.0001)
(defparameter *clipgrad-l1-default-threashold* 1.0)
						
(setf *read-default-float-format* *calculus-system*)

(defun change-calculus-system! (new-system)
  "WARNING: The function has side effects
  
   function for quickly changing the type of calculus"
   
  (setf *calculus-system* new-system)
  (setf *read-default-float-format* new-system)
  (setf magicl::*default-tensor-type* new-system))					
						
(defun ~= (x y &key (tolerance *approximately-equal-default-tolerance*))
  (<= (abs (- x y)) tolerance))		
  