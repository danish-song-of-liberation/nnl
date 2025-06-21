#| Philipp Mainlender (1841 - 1876)
   -
   Gott ist gestorben und sein 
   tod war das leben der welt. |#

(asdf:defsystem :nnl
  :license "MIT"
  :version "0.1.0"
  :homepage "https://github.com/danish-song-of-liberation/nnl"
  :description "Common lisp neural networks mvp framework"
  :depends-on (:fiveam :magicl)
  :serial nil
  :components ((:module "src"
				:components ((:module "magicl"
							  :components ((:file "magicl" :type "lisp")
										   (:file "coerce-data" :type "lisp")	
										   (:file "operations" :type "lisp")
										   (:file "deep-copy" :type "lisp")
										   (:file "random" :type "lisp")
										   (:file "broadcasting" :type "lisp")
									       (:file "high-level" :type "lisp")
										   (:file "numgrad" :type "lisp")))
										   
							 (:module "math"
							  :components ((:file "math" :type "lisp")
										   (:file "phi" :type "lisp")
										   
										   (:module "autodiff"
											:components ((:file "autodiff" :type "lisp")
														 (:file "autodiff-d" :type "lisp")
														 (:file "autodiff-i" :type "lisp")
														 (:file "autodiff-h" :type "lisp")))
														 
											(:file "basic-accessors" :type "lisp")
											(:file "broadcasting" :type "lisp")))
											
							 (:module "init"
							  :components ((:file "init" :type "lisp")
										   (:file "random" :type "lisp")
										   (:file "xavier-normal" :type "lisp")
										   (:file "xavier-uniform" :type "lisp")))	
											
							 (:module "models"
							  :components ((:file "models" :type "lisp")
										   (:file "fc" :type "lisp")
										   (:file "sequential" :type "lisp")))
														 
							 (:module "optims"
							  :components ((:file "optims" :type "lisp")
										   (:file "gd" :type "lisp")))
											
							 (:module "utils"
							  :components ((:file "utils" :type "lisp")
										   (:file "clip-grad" :type "lisp")))))))
																										
(defpackage :nnl.system
  (:use :cl)
  (:export :change-calculus-system!))		
						
(in-package :nnl.system)

(defparameter *calculus-system* 'single-float
  "Global variable that determines the floating-point 
   number system used for calculations within a specific 
   mathematical functionality")
   
(defparameter *leakyrelu-default-shift* 0.01)
(defparameter *clipgrad-l1-default-threashold* 1.0)
						
(setf *read-default-float-format* *calculus-system*)

(defun change-calculus-system! (new-system)
  "WARNING: The function has side effects
  
   function for quickly changing the type of calculus"
   
  (setf *calculus-system* new-system)
  (setf *read-default-float-format* new-system))					
						