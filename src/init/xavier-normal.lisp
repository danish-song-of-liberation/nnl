(in-package :nnl.init)

(defun xavier-normal-initialize (shapes sum-in-out &key (requiers-grad nil))
  (let* ((ceil (sqrt (/ 2 sum-in-out)))
		 (floor (- ceil)))
		 
	(random-initialize shapes :from floor :to ceil :requiers-grad requiers-grad)))
	