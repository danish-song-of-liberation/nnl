(defpackage :nnl.tests
  (:use :cl)
  (:export :run-all-tests :run-math-tests))
  
(in-package :nnl.tests)

(defun run-all-tests ()
  (fiveam:run-all-tests))

(defun run-math-tests ()
  (fiveam:run! 'nnl.math.tests::nnl.math-suite))  
  