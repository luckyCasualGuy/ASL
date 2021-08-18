from numpy import array, zeros, concatenate, ndarray

class MediapipeExtractor:
    
    def extract_landmarks(self, results) -> ndarray:
        result_landmarks = {
            'face': {
                'landmark': results.face_landmarks,
                'shape': (468,)
            },
            'left hand': {
                'landmark': results.left_hand_landmarks,
                'shape': (21,)
            },
            'right hand': {
                'landmark': results.right_hand_landmarks,
                'shape': (21,)
            },
            'pose': {
                'landmark': results.pose_landmarks,
                'shape': (33,)
            }
        }

        result_all = []
        result_temp = []
        for key, result in result_landmarks.items():
            if result['landmark']:
                result_temp = array([array([l.x, l.y, l.z]) for l in result['landmark'].landmark])
                result_temp = result_temp.flatten()
            else:
                result_temp = zeros((result['shape'][0] * 3, ))

            assert(result_temp.shape == (result['shape'][0] * 3, ))
            result_all.append(result_temp)
            
        shape = 0
        result_final = array([])
        for result in result_all:
            shape += result.shape[0]
            result_final = concatenate((result_final, result), axis=0)

        assert(result_final.shape == (shape,))

        return result_final