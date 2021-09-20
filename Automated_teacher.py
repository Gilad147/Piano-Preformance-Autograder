import random


def create_fake_teachers(number_of_teachers):
    teachers = []
    for i in range(number_of_teachers):
        pitch_unique_next_step = round(random.uniform(-0.05, 0), 2)
        rhythm_unique_next_step = round(random.uniform(-0.25, 0.25), 2)
        tempo_unique_next_step = round(random.uniform(-0.25, 0.25), 2)
        articulation_unique_next_step = round(random.uniform(-0.25, 0.25), 2)
        dynamics_unique_next_step = round(random.uniform(-0.3, 0.3), 2)

        pitch_unique_score = round(random.uniform(0.83, 0.88), 2)
        rhythm_unique_score = round(random.uniform(-0.1, 0.1), 2)
        tempo_unique_score = round(random.uniform(-0.1, 0.1), 2)
        articulation_unique_score = round(random.uniform(-0.1, 0.1), 2)
        dynamics_unique_score = round(random.uniform(-0.1, 0.1), 2)

        teacher_i = Teacher(pitch_unique_next_step, rhythm_unique_next_step, tempo_unique_next_step,
                            articulation_unique_next_step, dynamics_unique_next_step,
                            pitch_unique_score, rhythm_unique_score, tempo_unique_score,
                            articulation_unique_score, dynamics_unique_score)
        teachers.append(teacher_i)

    return teachers


def fake_teachers_feedback(performance, teachers, pitch_tech_score, tempo_tech_score, rhythm_tech_score,
                           articulation_tech_score,
                           dynamics_tech_score):
    for teacher in teachers:
        performance.teachers_grades.append([teacher.give_scores(pitch_tech_score, "Pitch"),
                                            teacher.give_scores(tempo_tech_score, "Tempo"),
                                            teacher.give_scores(rhythm_tech_score, "Rhythm"),
                                            teacher.give_scores([articulation_tech_score, dynamics_tech_score],
                                                                "Articulation & Dynamics"),
                                            teacher.give_scores([pitch_tech_score, tempo_tech_score, rhythm_tech_score,
                                                                 articulation_tech_score, dynamics_tech_score],
                                                                "Overall"),  # overall score
                                            teacher.give_next_step_recco(rhythm_score=rhythm_tech_score,
                                                                                dynamics_score=dynamics_tech_score,
                                                                                articulation_score=articulation_tech_score,
                                                                                pitch_score=pitch_tech_score,
                                                                                tempo_score=tempo_tech_score)])


class Teacher:
    def __init__(self, pitch_unique_next_step, rhythm_unique_next_step, tempo_unique_next_step, articulation_unique_next_step,
                 dynamics_unique_next_step,
                 pitch_unique_score, rhythm_unique_score, tempo_unique_score, articulation_unique_score,
                 dynamics_unique_score):
        self.pitch_unique_next_step = pitch_unique_next_step
        self.rhythm_unique_next_step = rhythm_unique_next_step
        self.tempo_unique_next_step = tempo_unique_next_step
        self.articulation_unique_next_step = articulation_unique_next_step
        self.dynamics_unique_next_step = dynamics_unique_next_step

        self.pitch_unique_score = pitch_unique_score
        self.rhythm_unique_score = rhythm_unique_score
        self.tempo_unique_score = tempo_unique_score
        self.articulation_unique_score = articulation_unique_score
        self.dynamics_unique_score = dynamics_unique_score

    def give_next_step_recco(self, pitch_score, rhythm_score, tempo_score, articulation_score, dynamics_score):
        """
        mapping:
            same piece = 0-2:
                0 = slower
                1 = same pace
                2 = faster
            new piece = 3-5
                3 = easier
                4 = same level
                5 = harder
        """

        pitch_score += self.pitch_unique_next_step
        rhythm_score += self.rhythm_unique_next_step
        tempo_score += self.tempo_unique_next_step
        articulation_score += self.articulation_unique_next_step
        dynamics_score += self.dynamics_unique_next_step

        if pitch_score < 0.75:
            return "3"
        if 0.75 <= pitch_score < 0.83:
            return "0"
        if 0.83 <= pitch_score < 0.93:
            if 0 < rhythm_score < 0.5:
                return "0"
            if 0.85 < rhythm_score:
                return "2"
            if 0.25 < tempo_score < 0.5 or dynamics_score < 0.5 or articulation_score < 0.3:
                return "0"
            if 0.5 <= articulation_score < 0.7 or \
                    0.75 <= tempo_score < 0.9 or dynamics_score < 0.7:
                return "1"
            else:
                return "4"
        else:
            cnt = 0
            if rhythm_score < 0.75:
                cnt += 1
            if tempo_score < 0.75:
                cnt += 1
            if dynamics_score < 0.5:
                cnt += 1
            if articulation_score < 0.75:
                cnt += 1
            if cnt >= 3:
                return "4"
            else:
                return "5"

    def give_scores(self, feature_score, feature_name):

        if feature_name == "Pitch":
            feature_score *= self.pitch_unique_score
        elif feature_name == "Rhythm":
            feature_score += self.rhythm_unique_score
        elif feature_name == "Tempo":
            feature_score += self.tempo_unique_score
        elif feature_name == "Articulation & Dynamics":
            feature_score[0] += self.articulation_unique_score
            feature_score[1] += self.dynamics_unique_score
            feature_score = sum(feature_score) / 2
        elif feature_name == "Overall":
            feature_score = \
                feature_score[0] * (0.1 + self.pitch_unique_score) * 0.5 + (
                        feature_score[1] + self.tempo_unique_score) * 0.05 + \
                (feature_score[2] + self.rhythm_unique_score) * 0.25 + \
                (feature_score[3] + self.articulation_unique_score) * 0.15 + feature_score[4] * 0.05
            return str(round(feature_score * 10))

        if feature_score < 0.3:
            return "0"
        elif 0.3 <= feature_score < 0.5:
            return "1"
        elif 0.5 <= feature_score < 0.7:
            return "2"
        elif 0.7 <= feature_score < 0.85:
            return "3"
        else:
            return "4"
