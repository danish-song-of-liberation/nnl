(defpackage :nnl.tests
  (:use :cl)
  (:export :run-all-tests :run-math-tests :run-init-tests))
  
(in-package :nnl.tests)

(defun run-all-tests ()
  (unless nnl.system::*silent-mode*
    (warn "~%Please, write `(setf *random-state* (make-random-state t))` before running the tests.~%this is optional but highly recommended for accurate results.~%you can also omit this and write `(setf nnl.system::*silent* t)`~%"))

  (fiveam:run-all-tests))

(defun run-math-tests ()
  (fiveam:run! 'nnl.math.tests::nnl.math-suite))  
  
(defun run-init-tests ()
  (unless nnl.system::*silent-mode*
    (warn "~%Please, write `(setf *random-state* (make-random-state t))` before running the tests.~%this is optional but highly recommended for accurate results.~%you can also omit this and write `(setf nnl.system::*silent* t)`~%"))
  
  (fiveam:run! 'nnl.init.tests::nnl.init-suite))  
  