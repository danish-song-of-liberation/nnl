(in-package :nnl.init.tests)

(fiveam:in-suite nnl.init-suite)

;; TODO DOCSTRINGS 

(fiveam:test nnl.init-tests/random-shapes
  (let ((result-1 (nnl.init:random-initialize '(3 3)))
		(result-2 (nnl.init:random-initialize '(8)))
		(result-3 (nnl.init:random-initialize '(6 2 4)))
		(result-4 (nnl.init:random-initialize '(1 1 1))))
		
    (fiveam:is (equal (nnl.math:shape result-1) '(3 3)))
	(fiveam:is (equal (nnl.math:shape result-2) '(8)))
	(fiveam:is (equal (nnl.math:shape result-3) '(6 2 4)))
	(fiveam:is (equal (nnl.math:shape result-4) '(1 1 1)))))
  
(fiveam:test nnl.init-tests/random-range
  "todo rework magicl:tref to nnl.math:tref"

  (let ((result-1 (nnl.init:random-initialize '(8) :from -1.0 :to 1.0))
		(result-2 (nnl.init:random-initialize '(8) :from -0.1 :to 0.1))
		(result-3 (nnl.init:random-initialize '(8) :from -100 :to 100))
		(result-4 (nnl.init:random-initialize '(8) :from -0.5 :to 0.5)))
    
	(loop for i from 0 below 8
	
		  do (fiveam:is (<= -1.0 (magicl:tref (nnl.math.autodiff::data result-1) i) 1.0)) ; (<= -1.0 x 1.0) equivalent to (and (<= x -1.0) (>= x 1.0))
		  do (fiveam:is (<= -0.1 (magicl:tref (nnl.math.autodiff::data result-2) i) 0.1))
		  do (fiveam:is (<= -100 (magicl:tref (nnl.math.autodiff::data result-3) i) 100))
		  do (fiveam:is (<= -0.5 (magicl:tref (nnl.math.autodiff::data result-4) i) 0.5)))))
		  
(fiveam:test nnl.init-tests/random-requires-grad
  (let ((result-1 (nnl.init:random-initialize '(3 3) :requires-grad t))		
		(result-2 (nnl.init:random-initialize '(3 3) :requires-grad nil)))

	(fiveam:is-true (nnl.math:requires-grad result-1))
	(fiveam:is-false (nnl.math:requires-grad result-2))))
	
(fiveam:test nnl.init-tests/random-default-range 
  "todo rework magicl:tref to nnl.math:tref"
  
  (let ((result-1 (nnl.init:random-initialize '(6)))
		(result-2 (nnl.init:random-initialize '(6))))
		
	(loop for i from 0 below 5
		  
		  do (fiveam:is (<= -1.0 (magicl:tref (nnl.math.autodiff::data result-1) i) 1.0))
		  do (fiveam:is (<= -1.0 (magicl:tref (nnl.math.autodiff::data result-2) i) 1.0)))))
		  