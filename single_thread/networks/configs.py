DWM_DGL = {
    'resnet32': {
        '1': [],  # End-to-end
        '1A': [[3, 4]],  # detaching last FC from the network
        '2': [[2, 1], [3, 4]],
        '3': [[1, 4], [2, 4], [3, 4]],
        '4': [[1, 2], [2, 1], [3, 0], [3, 4]],
        '8': [[1, 0], [1, 2], [1, 4],
              [2, 1], [2, 3],
              [3, 0], [3, 2], [3, 4]],
        '16': [[0, 0],
               [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
               [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
               [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
    }
}

InfoPro = {
    'resnet32': {
        '1': [],  # End-to-end
        '2': [[2, 1]],
        '3': [[1, 4], [2, 4]],
        '4': [[1, 2], [2, 1], [3, 0]],
        '8': [[1, 0], [1, 2], [1, 4],
            [2, 1], [2, 3],
            [3, 0], [3, 2]],
        '16': [[0, 0],
             [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
             [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
             [3, 0], [3, 1], [3, 2], [3, 3]],
    },
    'resnet110': {
        '1': [],  # End-to-end
        '2': [[2, 8]],
        '3': [[1, 17], [2, 17]],
        '4': [[1, 11], [2, 7], [3, 3]],
        '8': [[1, 4], [1, 11],
            [2, 0], [2, 7], [2, 14],
            [3, 3], [3, 10]],
        '16': [[1, 1], [1, 4], [1, 7], [1, 10], [1, 13], [1, 16],
             [2, 1], [2, 4], [2, 7], [2, 11], [2, 15],
             [3, 1], [3, 5], [3, 9], [3, 13]],
    }
}

InfoPro_balanced_memory = {
    'resnet110': {
        'cifar10': {
            '1': [],  # End-to-end
            '2': [[1, 14]],
            '3': [[1, 9], [2, 2]],
            '4': [[1, 6], [1, 14], [2, 8]],
        },
        'stl10': {
            '1': [],  # End-to-end
            '2': [[1, 14]],
            '3': [[1, 8], [2, 1]],
            '4': [[1, 6], [1, 14], [2, 8]],
        }
    }
}
