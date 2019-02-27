import cv2
import numpy as np
from numba import vectorize

# Cpu version
def vector_add_cpu(a, b, num_elements):
    c = np.zeros( num_elements, dtype=np.float32 )
    for i in range( num_elements ):
        c[i] = a[i] + b[i]
    return c

# Gpu version
@vectorize(["float32(float32, float32)"], target='cuda')
def vector_add_gpu(a, b):
    return a + b

def main(num_elements=1000000):
    a_src = np.ones(num_elements, dtype=np.float32)
    b_src = np.ones(num_elements, dtype=np.float32)

    # Time cpu
    t0 = cv2.getTickCount()
    vector_add_cpu(a_src, b_src, num_elements)
    t1 = cv2.getTickCount()
    print('CPU clock-cycles: ', (t1-t0), 'Time: ', (t1-t0)/cv2.getTickFrequency())

    # Time gpu
    t0 = cv2.getTickCount()
    vector_add_gpu(a_src, b_src)
    t1 = cv2.getTickCount()
    print('GPU clock-cycles: ', (t1-t0), 'Time: ', (t1-t0)/cv2.getTickFrequency())

    return 0

if __name__ == '__main__':
    main()