(in-package :nnl.init.tests)

(fiveam:in-suite nnl.init-suite)

;; TODO DOCSTRINGS 

(fiveam:test nnl.init-tests/xavier-normal-shapes
  (let ((result-1 (nnl.init:xavier-normal-initialize '(4 7) 1))
		(result-2 (nnl.init:xavier-normal-initialize '(10) 1))
		(result-3 (nnl.init:xavier-normal-initialize '(2 3 4) 1))
		(result-4 (nnl.init:xavier-normal-initialize '(1 1 1) 1)))
		
    (fiveam:is (equal (nnl.math:shape result-1) '(4 7)))
	(fiveam:is (equal (nnl.math:shape result-2) '(10)))
	(fiveam:is (equal (nnl.math:shape result-3) '(2 3 4)))
	(fiveam:is (equal (nnl.math:shape result-4) '(1 1 1)))))
	
(fiveam:test nnl.init-suite/xavier-normal-range
  (let* ((sum-out-in-1 16)
		 (sum-out-in-2 24)
		 (sum-out-in-3 5)
		 (sum-out-in-4 128)
		 (result-1 (nnl.init:xavier-normal-initialize '(8) sum-out-in-1))
		 (result-2 (nnl.init:xavier-normal-initialize '(8) sum-out-in-2))
		 (result-3 (nnl.init:xavier-normal-initialize '(8) sum-out-in-3))
		 (result-4 (nnl.init:xavier-normal-initialize '(8) sum-out-in-4))
		 (ceil-1 (sqrt (/ 2 sum-out-in-1)))
		 (ceil-2 (sqrt (/ 2 sum-out-in-2)))
		 (ceil-3 (sqrt (/ 2 sum-out-in-3)))
		 (ceil-4 (sqrt (/ 2 sum-out-in-4)))
		 (floor-1 (- ceil-1))
		 (floor-2 (- ceil-2))
		 (floor-3 (- ceil-3))
		 (floor-4 (- ceil-4)))
		 
	(loop for i from 0 below 8
	
		  do (fiveam:is (<= floor-1 (magicl:tref (nnl.math.autodiff::data result-1) i) ceil-1)) ; (<= -1.0 x 1.0) equivalent to (and (<= x -1.0) (>= x 1.0))
		  do (fiveam:is (<= floor-2 (magicl:tref (nnl.math.autodiff::data result-2) i) ceil-2))
		  do (fiveam:is (<= floor-3 (magicl:tref (nnl.math.autodiff::data result-3) i) ceil-3))
		  do (fiveam:is (<= floor-4 (magicl:tref (nnl.math.autodiff::data result-4) i) ceil-4)))))
		  
(fiveam:test nnl.init-suite/xavier-normal-requires-grad
  (let ((result-1 (nnl.init:xavier-normal-initialize '(3 3) 1 :requires-grad t))		
		(result-2 (nnl.init:xavier-normal-initialize '(3 3) 1 :requires-grad nil)))

	(fiveam:is-true (nnl.math:requires-grad result-1))
	(fiveam:is-false (nnl.math:requires-grad result-2))))	  
	