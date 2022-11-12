import json
import tensorflow.compat.v1 as tf


# Inputs
# -----------------------------------------------------------------------------
inpfile = './inpfile.json'
mem_fraction = 0.3
growth = True
num_threads = 8


# Main
# -----------------------------------------------------------------------------
def main():
    inpdata = json.load(open(inpfile, 'r'))
    inpdata['config'] = create_serialized_options(
        mem_fraction, growth, num_threads
    )
    json.dump(inpdata, open(inpfile, 'w'), indent=4)

def create_serialized_options(mem_fraction, growth, num_threads):
    config = tf.ConfigProto()
    # Allocate GPU memory
    config.gpu_options.per_process_gpu_memory_fraction = mem_fraction
    config.gpu_options.allow_growth = growth
    # Set number of threads
    config.inter_op_parallelism_threads = num_threads
    config.intra_op_parallelism_threads = num_threads
    # Serialize
    serialized = config.SerializeToString()
    return list(map(hex, serialized))


if __name__ == '__main__':
    main()