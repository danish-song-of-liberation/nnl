(in-package :nnl.math.tests)

(fiveam:in-suite nnl.math-suite)

;; TODO DOCSTRINGS 

(fiveam:test nnl.math-test/relu-basic
  (fiveam:is (= 2 (nnl.math:relu 2)))
  (fiveam:is (= 0 (nnl.math:relu -3)))
  (fiveam:is (= 1 (nnl.math:relu 1)))
  (fiveam:is (= 0 (nnl.math:relu 0))))
  
(fiveam:test nnl.math-test/relu-derivative
  (fiveam:is (= 1 (nnl.math:relu 3 :derivative t)))
  (fiveam:is (= 0 (nnl.math:relu -2 :derivative t)))
  (fiveam:is (= 1 (nnl.math:relu 7 :derivative t)))
  (fiveam:is (= 0 (nnl.math:relu -11 :derivative t))))
  
(fiveam:test nnl.math-test/leaky-relu-basic
  (fiveam:is (= 1 (nnl.math:leaky-relu 1)))
  (fiveam:is (= 3 (nnl.math:leaky-relu 3)))
  (fiveam:is (= 0 (nnl.math:leaky-relu 0)))
  (fiveam:is (nnl.system:~= (* -5 nnl.system::*leakyrelu-default-shift*) (nnl.math:leaky-relu -5) :tolerance 0.1)))  
  
(fiveam:test nnl.math-test/leaky-relu-derivative
  (fiveam:is (= 1 (nnl.math:leaky-relu 2 :derivative t)))
  (fiveam:is (= nnl.system::*leakyrelu-default-shift* (nnl.math:leaky-relu -4 :derivative t :shift nnl.system::*leakyrelu-default-shift*)))
  (fiveam:is (= nnl.system::*leakyrelu-default-shift* (nnl.math:leaky-relu 0 :derivative t)))
  (fiveam:is (= 1 (nnl.math:leaky-relu 1 :derivative t))))  
  
(fiveam:test nnl.math-test/sigmoid-basic
  (fiveam:is (= 0.5 (nnl.math:sigmoid 0)))
  (fiveam:is (nnl.system:~= 0.623 (nnl.math:sigmoid 0.5) :tolerance 0.1))
  (fiveam:is (nnl.system:~= 0.183 (nnl.math:sigmoid -1.5) :tolerance 0.1))
  (fiveam:is (nnl.system:~= 0.731 (nnl.math:sigmoid 1.0) :tolerance 0.1)))  
  
(fiveam:test nnl.math-test/sigmoid-derivative
  (fiveam:is (= 0.25 (nnl.math:sigmoid 0 :derivative t)))
  (fiveam:is (nnl.system:~= 0.196 (nnl.math:sigmoid -1 :derivative t) :tolerance 0.1))
  (fiveam:is (nnl.system:~= 0.007 (nnl.math:sigmoid 2.5 :derivative t) :tolerance 0.1))
  (fiveam:is (nnl.system:~= 0.104 (nnl.math:sigmoid -2 :derivative t) :tolerance 0.1)))
  
(fiveam:test nnl.math-test/tanh-basic
  (fiveam:is (= 0.0 (nnl.math:tanh 0)))
  (fiveam:is (nnl.system:~= 0.761 (nnl.math:tanh 1) :tolerance 0.1))
  (fiveam:is (nnl.system:~= -0.462 (nnl.math:tanh -0.5) :tolerance 0.1))
  (fiveam:is (nnl.system:~= -0.995 (nnl.math:tanh -3.0) :tolerance 0.1)))

(fiveam:test nnl.math-test/tanh-derivative
  (fiveam:is (= 1.0 (nnl.math:tanh 0 :derivative t)))
  (fiveam:is (nnl.system:~= 0.419 (nnl.math:tanh 1 :derivative t) :tolerance 0.1))
  (fiveam:is (nnl.system:~= 0.419 (nnl.math:tanh -1 :derivative t) :tolerance 0.1))
  (fiveam:is (nnl.system:~= 0.180 (nnl.math:tanh -1.5 :derivative t) :tolerance 0.1)))
  