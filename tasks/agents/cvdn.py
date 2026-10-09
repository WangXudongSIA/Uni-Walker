from .mp3d_agent import MP3DAgent

class CVDNAgent(MP3DAgent):
    name = "cvdn"

    def get_prompt(self, task, *args, **kwargs):
        if task == 'navigation':
            return self.get_navigation_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_navigation_prompt(self, instruction, hist_num, cand_num, cls_token):
        prompt = 'You are a navigation agent. You are required to understand the dialog to find the described room and go to the room.\n'
        # Task
        prompt += '### Dialog: {} \n'.format(instruction)
        # Observation
        prompt += 'The current observation contains several directions you can go, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # History
        prompt += 'You may also need the navigation history, which contains the images corresponding to your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### The history: {}\n'.format(hist_text)
        # Output Hint
        prompt += 'Based on observation and history to infer where is the current location, and then select the next direction from the candidates to go to the target location.\n'
        prompt += '### Output: {}'.format(cls_token)
        
        return prompt
