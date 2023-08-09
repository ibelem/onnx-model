'''
ONNX model required ops, and data types for the ops.
'''
import os
import argparse
import onnx
import onnxruntime

ONNX_DATA_TYPE_ID = {
    'UNDEFINED': 0,
    'FLOAT': 1,
    'UINT8': 2,
    'INT8': 3,
    'UINT16': 4,
    'INT16': 5,
    'INT32': 6,
    'INT64': 7,
    'STRING': 8,
    'BOOL': 9,
    'FLOAT16': 10,
    'DOUBLE': 11,
    'UINT32': 12,
    'UINT64': 13,
    'COMPLEX64': 14,
    'COMPLEX128': 15,
    'BFLOAT16': 16,
    'FLOAT8E4M3FN': 17,
    'FLOAT8E4M3FNUZ': 18,
    'FLOAT8E5M2': 19,
    'FLOAT8E5M2FNUZ': 20
}

ONNX_DATA_TYPE = {
    0: 'undefined',
    1: 'float32',
    2: 'uint8',
    3: 'int8',
    4: 'uint16',
    5: 'int16',
    6: 'int32',
    7: 'int64',
    8: 'string',
    9: 'bool',
    10: 'float16',
    11: 'double',
    12: 'uint32',
    13: 'uint64',
    14: 'COMPLEX64',
    15: 'COMPLEX128',
    16: 'bfloat16',
    17: 'FLOAT8E4M3FN',
    18: 'FLOAT8E4M3FNUZ',
    19: 'FLOAT8E5M2',
    20: 'FLOAT8E5M2FNUZ'
}

ONNX_ATTRIBUTE_TYPE_ID = {
    0: 'UNDEFINED',
    1: 'FLOAT',
    2: 'INT',
    3: 'STRING',
    4: 'TENSOR',
    5: 'GRAPH',
    6: 'FLOATS',
    7: 'INTS',
    8: 'STRINGS',
    9: 'TENSORS',
    10: 'GRAPHS',
    11: 'SPARSE_TENSOR',
    12: 'SPARSE_TENSORS',
    13: 'TYPE_PROTO',
    14: 'TYPE_PROTOS'
}

ONNX_ATTRIBUTE_TYPE = {
    'UNDEFINED': 'undefined',
    'FLOAT': 'float32',
    'INT': 'int64',
    'STRING': 'string',
    'TENSOR': 'tensor',
    'GRAPH': 'graph',
    'FLOATS': 'float32[]',
    'INTS': 'int64[]',
    'STRINGS': 'string[]',
    'TENSORS': 'tensor[]',
    'GRAPHS': 'graph',
    'SPARSE_TENSOR': 'tensor',
    'SPARSE_TENSORS': 'tensor[]',
    'TYPE_PROTO': 'type',
    'TYPE_PROTOS': 'type[]'
}

def clear_file(model: str):
    '''Clear file before writing'''
    with open(os.path.basename(model) + '.txt', 'w', encoding='UTF-8') as file:
        pass

    # # Do no run following code for large models
    # with open(os.path.basename(model) + '_raw.txt', 'w', encoding='UTF-8') as file:
    #     onnx_model = onnx.load(model)
    #     file.write(str(onnx_model.graph))

def get_nodes(model: str):
    '''Get ops of onnx models'''
    onnx_model = onnx.load(model)
    onnx_nodes = set()
    for node in onnx_model.graph.node:
        if node.op_type:
            onnx_nodes.add(node.op_type)
    onnx_nodes = sorted(list(onnx_nodes))

    with open(os.path.basename(model) + '.txt', 'a', encoding='UTF-8') as file:
        print('--------------------------------')
        file.write('--------------------------------\n')
        print(onnx_nodes)
        file.write(str(onnx_nodes) + '\n')
        print('Total: ' + str(len(onnx_nodes)))
        file.write('Total: ' + str(len(onnx_nodes)) + '\n')

def get_node_data_type(model: str):
    '''Get ops of onnx models'''
    onnx_model = onnx.load(model)

    onnx_input = {}
    onnx_output = {}

    for initializer in onnx_model.graph.initializer:
        onnx_input[initializer.name] = initializer.data_type

    for input_info in onnx_model.graph.input:
        onnx_input[input_info.name] = input_info.type.tensor_type.elem_type

    for value_info in onnx_model.graph.value_info:
        onnx_output[value_info.name] = value_info.type.tensor_type.elem_type

    for output_info in onnx_model.graph.output:
        onnx_output[output_info.name] = output_info.type.tensor_type.elem_type

    with open(os.path.basename(model) + '.txt', 'a', encoding='UTF-8') as file:
        for node in onnx_model.graph.node:
            if node.op_type and node.op_type != 'Constant':
                print('--------------------------------')
                file.write('--------------------------------\n')
                print(node.op_type)
                file.write(node.op_type + '\n')
                for node_input in node.input:
                    if onnx_input.get(node_input):
                        print('  input ' +
                              ONNX_DATA_TYPE[onnx_input.get(node_input)])
                        file.write(
                            '  input ' + ONNX_DATA_TYPE[onnx_input.get(node_input)] + '\n')
                for node_output in node.output:
                    if onnx_output.get(node_output):
                        print('  output ' +
                              ONNX_DATA_TYPE[onnx_output.get(node_output)])
                        file.write(
                            '  output ' + ONNX_DATA_TYPE[onnx_output.get(node_output)] + '\n')
                for attr in node.attribute:
                    print('  attribute ' + attr.name + ' ' +
                          ONNX_ATTRIBUTE_TYPE[ONNX_ATTRIBUTE_TYPE_ID[attr.type]])
                    file.write('  attribute ' + attr.name + ' ' +
                               ONNX_ATTRIBUTE_TYPE[ONNX_ATTRIBUTE_TYPE_ID[attr.type]] + '\n')


def get_input_output(model: str):
    '''Get input / output data type of onnx models'''
    sess = onnxruntime.InferenceSession(model)

    input_data_types = {}
    for input_node in sess.get_inputs():
        input_data_types[input_node.name] = input_node.type

    # Get the data type of output nodes
    output_data_types = {}
    for output_node in sess.get_outputs():
        output_data_types[output_node.name] = output_node.type

    print('input ' + str(input_data_types))
    print('output ' + str(output_data_types))

    with open(os.path.basename(model) + '.txt', 'a', encoding='UTF-8') as file:
        file.write('input ' + str(input_data_types) + '\n')
        file.write('output ' + str(output_data_types)+ '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='', required=True, type=str)
    args = parser.parse_args()
    clear_file(args.model)
    print(args.model)
    print('--------------------------------')
    get_input_output(args.model)
    get_nodes(args.model)
    get_node_data_type(args.model)