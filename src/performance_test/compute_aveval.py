
if __name__ == '__main__':
    file_name = 'tllmtorch_tllmtorch.res'
    visual_durations = []
    llm_durations = []
    e2e_durations = []
    new_tokens = []
    with open(file_name, 'r') as f:
        for line in f:
            if 'Visual encoder duration' in line:
                visual_durations.append(float(line.split(' ')[3]))
            elif 'Language model duration' in line:
                llm_durations.append(float(line.split(' ')[3]))
            elif 'E2E duration' in line:
                e2e_durations.append(float(line.split(' ')[2]))
            elif 'New tokens' in line:
                new_tokens.append(int(line.split(' ')[2]))
    assert len(visual_durations) == len(llm_durations) == len(e2e_durations) == len(new_tokens)
    print(f'We totally got {len(visual_durations)} results with {new_tokens[0]} tokens.')
    print(f"Average Visual Encoder Duration is :{sum(visual_durations)/len(visual_durations)} s")
    print(f"Average Language Model Duration is :{sum(llm_durations)/sum(new_tokens)} s")
    print(f"Average E2E Duration is :{sum(e2e_durations)/len(e2e_durations)} s")
        