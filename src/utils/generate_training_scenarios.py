"""
Curriculum basé sur l'interpolation entre les 5 scénarios prédéfinis.

Objectif: 
- Pas d'overfitting sur un scénario spécifique
- Générer des environnements "entre" les scénarios du prof
- Progression smooth de facile à difficile
"""

import numpy as np
from typing import Dict, Tuple, List


# Les 5 scénarios du prof (extraits de __init__.py)
PROFESSOR_SCENARIOS = {
    'static_headwind': {
        'wind_init_params': {
            'base_speed': 3.0,
            'base_direction': (0.0, -1.0),  # Pure North wind (headwind)
            'pattern_scale': 64,
            'pattern_strength': 0.2,
            'strength_variation': 0.15,
            'noise': 0.08
        },
        'wind_evol_params': {
            'wind_change_prob': 0.0,
            'pattern_scale': 128,
            'perturbation_angle_amplitude': 0.0,
            'perturbation_strength_amplitude': 0.0,
            'rotation_bias': 0.0,
            'bias_strength': 0.0
        }
    },
    'training_1': {  # NNW wind
        'wind_init_params': {
            'base_speed': 3.0,
            'base_direction': (-0.8, -0.2),
            'pattern_scale': 32,
            'pattern_strength': 0.3,
            'strength_variation': 0.4,
            'noise': 0.1
        },
        'wind_evol_params': {
            'wind_change_prob': 1.0,
            'pattern_scale': 128,
            'perturbation_angle_amplitude': 0.1,
            'perturbation_strength_amplitude': 0.1,
            'rotation_bias': 0.02,
            'bias_strength': 1.0
        }
    },
    'training_2': {  # SSW wind
        'wind_init_params': {
            'base_speed': 3.0,
            'base_direction': (-0.2, 0.8),
            'pattern_scale': 128,
            'pattern_strength': 0.6,
            'strength_variation': 0.3,
            'noise': 0.1
        },
        'wind_evol_params': {
            'wind_change_prob': 1.0,
            'pattern_scale': 128,
            'perturbation_angle_amplitude': 0.1,
            'perturbation_strength_amplitude': 0.1,
            'rotation_bias': 0.02,
            'bias_strength': 1.0
        }
    },
    'training_3': {  # NNE wind
        'wind_init_params': {
            'base_speed': 3.0,
            'base_direction': (0.2, -0.8),
            'pattern_scale': 32,
            'pattern_strength': 0.4,
            'strength_variation': 0.2,
            'noise': 0.1
        },
        'wind_evol_params': {
            'wind_change_prob': 1.0,
            'pattern_scale': 128,
            'perturbation_angle_amplitude': 0.1,
            'perturbation_strength_amplitude': 0.1,
            'rotation_bias': 0.02,
            'bias_strength': 1.0
        }
    },
    'simple_static': {  # NE wind stable
        'wind_init_params': {
            'base_speed': 3.0,
            'base_direction': (-0.7, -0.7),
            'pattern_scale': 32,
            'pattern_strength': 0.1,
            'strength_variation': 0.1,
            'noise': 0.05
        },
        'wind_evol_params': {
            'wind_change_prob': 0.0,
            'pattern_scale': 128,
            'perturbation_angle_amplitude': 0.0,
            'perturbation_strength_amplitude': 0.0,
            'rotation_bias': 0.0,
            'bias_strength': 0.0
        }
    }
}


def normalize_direction(direction: Tuple[float, float]) -> Tuple[float, float]:
    """Normalise un vecteur direction."""
    dx, dy = direction
    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


def interpolate_params(params1: Dict, params2: Dict, alpha: float) -> Dict:
    """
    Interpole linéairement entre deux dictionnaires de paramètres.
    
    Args:
        params1: Premier dictionnaire
        params2: Deuxième dictionnaire
        alpha: Coefficient d'interpolation (0 = params1, 1 = params2)
    
    Returns:
        Dictionnaire interpolé
    """
    result = {}
    
    for key in params1.keys():
        val1 = params1[key]
        val2 = params2[key]
        
        if isinstance(val1, tuple):
            # Direction : interpoler puis normaliser
            interp_x = (1 - alpha) * val1[0] + alpha * val2[0]
            interp_y = (1 - alpha) * val1[1] + alpha * val2[1]
            result[key] = normalize_direction((interp_x, interp_y))
        else:
            # Scalaire : interpolation linéaire
            result[key] = (1 - alpha) * val1 + alpha * val2
    
    return result


def sample_scenario_mix(scenario_weights: Dict[str, float]) -> Dict:
    """
    Mélange plusieurs scénarios avec des poids.
    
    Args:
        scenario_weights: Dictionnaire {scenario_name: weight}
                         Ex: {'static_headwind': 0.7, 'training_1': 0.3}
    
    Returns:
        Paramètres de vent interpolés
    """
    # Normaliser les poids
    total_weight = sum(scenario_weights.values())
    normalized_weights = {k: v/total_weight for k, v in scenario_weights.items()}
    
    # Calculer la moyenne pondérée des paramètres
    wind_init_params = {}
    wind_evol_params = {}
    
    # Initialiser avec zéros
    first_scenario = list(PROFESSOR_SCENARIOS.values())[0]
    for key in first_scenario['wind_init_params'].keys():
        if isinstance(first_scenario['wind_init_params'][key], tuple):
            wind_init_params[key] = (0.0, 0.0)
        else:
            wind_init_params[key] = 0.0
    
    for key in first_scenario['wind_evol_params'].keys():
        wind_evol_params[key] = 0.0
    
    # Sommer les contributions pondérées
    for scenario_name, weight in normalized_weights.items():
        scenario = PROFESSOR_SCENARIOS[scenario_name]
        
        for key, val in scenario['wind_init_params'].items():
            if isinstance(val, tuple):
                wind_init_params[key] = (
                    wind_init_params[key][0] + weight * val[0],
                    wind_init_params[key][1] + weight * val[1]
                )
            else:
                wind_init_params[key] += weight * val
        
        for key, val in scenario['wind_evol_params'].items():
            wind_evol_params[key] += weight * val
    
    # Normaliser la direction
    wind_init_params['base_direction'] = normalize_direction(
        wind_init_params['base_direction']
    )
    
    return wind_init_params, wind_evol_params


class ScenarioInterpolationCurriculum:
    """
    Curriculum qui interpole entre les 5 scénarios du prof.
    
    Stratégie par phases:
    1. Phase Easy: Majoritairement static_headwind + simple_static
    2. Phase Medium: Mix équilibré des 5 scénarios
    3. Phase Hard: Interpolations aléatoires complexes
    4. Phase Mixed: Tout le spectre
    """
    
    def __init__(self, total_episodes: int = 144000):
        self.total_episodes = total_episodes
        
        # Définir les phases
        self.stages = [
            {
                'name': 'Static Training',
                'episodes': (0, 15000),
                'strategy': 'mostly_static',
                'recall_prob': 0.5,  # 50% de pure static
            },
            {
                'name': 'Gentle Dynamics',
                'episodes': (15000, 30000),
                'strategy': 'static_to_dynamic',
                'recall_prob': 0.3,
            },
            {
                'name': 'Full Scenarios',
                'episodes': (30000, 72000),
                'strategy': 'balanced_scenarios',
                'recall_prob': 0.2,
            },
            {
                'name': 'Interpolation Training',
                'episodes': (72000, 96000),
                'strategy': 'pairwise_interpolation',
                'recall_prob': 0.15,
            },
            {
                'name': 'Complex Mix',
                'episodes': (96000, 120000),
                'strategy': 'weighted_mix',
                'recall_prob': 0.1,
            },
            {
                'name': 'Full Spectrum',
                'episodes': (120000, 144000),
                'strategy': 'random_all',
                'recall_prob': 0.15,
            },
        ]
    
    def get_current_stage(self, episode: int) -> Dict:
        """Retourne la phase actuelle."""
        for stage in self.stages:
            start, end = stage['episodes']
            if start <= episode < end:
                return stage
        return self.stages[-1]
    
    def sample_params(self, episode: int) -> Tuple[Dict, Dict, str]:
        """
        Sample des paramètres selon l'épisode courant.
        
        Returns:
            wind_init_params, wind_evol_params, description
        """
        stage = self.get_current_stage(episode)
        strategy = stage['strategy']
        recall_prob = stage['recall_prob']
        
        # Recall: revenir à un scénario pur simple
        if np.random.random() < recall_prob:
            scenario_name = np.random.choice(['static_headwind', 'simple_static'])
            scenario = PROFESSOR_SCENARIOS[scenario_name]
            return (
                scenario['wind_init_params'].copy(),
                scenario['wind_evol_params'].copy(),
                f"Recall: {scenario_name}"
            )
        
        # Stratégies selon la phase
        if strategy == 'mostly_static':
            return self._sample_mostly_static()
        
        elif strategy == 'static_to_dynamic':
            return self._sample_static_to_dynamic()
        
        elif strategy == 'balanced_scenarios':
            return self._sample_balanced_scenarios()
        
        elif strategy == 'pairwise_interpolation':
            return self._sample_pairwise_interpolation()
        
        elif strategy == 'weighted_mix':
            return self._sample_weighted_mix()
        
        else:  # random_all
            return self._sample_random_all()
    
    def _sample_mostly_static(self):
        """Phase 1: Majoritairement statique."""
        if np.random.random() < 0.7:
            # 70% static headwind ou simple static
            scenario_name = np.random.choice(['static_headwind', 'simple_static'])
        else:
            # 30% interpolation entre les deux static
            alpha = np.random.uniform(0, 1)
            init1 = PROFESSOR_SCENARIOS['static_headwind']['wind_init_params']
            init2 = PROFESSOR_SCENARIOS['simple_static']['wind_init_params']
            evol1 = PROFESSOR_SCENARIOS['static_headwind']['wind_evol_params']
            evol2 = PROFESSOR_SCENARIOS['simple_static']['wind_evol_params']
            
            return (
                interpolate_params(init1, init2, alpha),
                interpolate_params(evol1, evol2, alpha),
                f"Interpolation: static_headwind ↔ simple_static (α={alpha:.2f})"
            )
        
        scenario = PROFESSOR_SCENARIOS[scenario_name]
        return (
            scenario['wind_init_params'].copy(),
            scenario['wind_evol_params'].copy(),
            f"Pure: {scenario_name}"
        )
    
    def _sample_static_to_dynamic(self):
        """Phase 2: Transition du statique vers le dynamique."""
        # Choisir un scénario static et un dynamic
        static_name = np.random.choice(['static_headwind', 'simple_static'])
        dynamic_name = np.random.choice(['training_1', 'training_2', 'training_3'])
        
        # Alpha croît avec la progression dans cette phase
        # Au début: plus de static, à la fin: plus de dynamic
        alpha = np.random.uniform(0.3, 0.7)  # Transition graduelle
        
        init1 = PROFESSOR_SCENARIOS[static_name]['wind_init_params']
        init2 = PROFESSOR_SCENARIOS[dynamic_name]['wind_init_params']
        evol1 = PROFESSOR_SCENARIOS[static_name]['wind_evol_params']
        evol2 = PROFESSOR_SCENARIOS[dynamic_name]['wind_evol_params']
        
        return (
            interpolate_params(init1, init2, alpha),
            interpolate_params(evol1, evol2, alpha),
            f"Static→Dynamic: {static_name} → {dynamic_name} (α={alpha:.2f})"
        )
    
    def _sample_balanced_scenarios(self):
        """Phase 3: Mix équilibré des 5 scénarios."""
        # Choisir un scénario aléatoire parmi les 5
        scenario_name = np.random.choice(list(PROFESSOR_SCENARIOS.keys()))
        scenario = PROFESSOR_SCENARIOS[scenario_name]
        
        # Ajouter un peu de bruit pour éviter l'overfitting exact
        wind_init = scenario['wind_init_params'].copy()
        wind_evol = scenario['wind_evol_params'].copy()
        
        # Petit bruit sur les paramètres (±10%)
        for key, val in wind_init.items():
            if isinstance(val, tuple):
                noise_x = np.random.uniform(-0.1, 0.1)
                noise_y = np.random.uniform(-0.1, 0.1)
                wind_init[key] = normalize_direction((val[0] + noise_x, val[1] + noise_y))
            elif key != 'base_speed':  # Ne pas toucher à la vitesse
                wind_init[key] = np.clip(val * np.random.uniform(0.9, 1.1), 0, 2)
        
        return (wind_init, wind_evol, f"Scenario: {scenario_name} (noisy)")
    
    def _sample_pairwise_interpolation(self):
        """Phase 4: Interpolation entre paires de scénarios."""
        # Choisir 2 scénarios aléatoires
        scenarios = list(PROFESSOR_SCENARIOS.keys())
        scenario1, scenario2 = np.random.choice(scenarios, size=2, replace=False)
        
        # Alpha uniforme
        alpha = np.random.uniform(0, 1)
        
        init1 = PROFESSOR_SCENARIOS[scenario1]['wind_init_params']
        init2 = PROFESSOR_SCENARIOS[scenario2]['wind_init_params']
        evol1 = PROFESSOR_SCENARIOS[scenario1]['wind_evol_params']
        evol2 = PROFESSOR_SCENARIOS[scenario2]['wind_evol_params']
        
        return (
            interpolate_params(init1, init2, alpha),
            interpolate_params(evol1, evol2, alpha),
            f"Pair: {scenario1} ↔ {scenario2} (α={alpha:.2f})"
        )
    
    def _sample_weighted_mix(self):
        """Phase 5: Mélange pondéré de 2-3 scénarios."""
        # Choisir 2 ou 3 scénarios
        num_scenarios = np.random.choice([2, 3])
        scenarios = list(PROFESSOR_SCENARIOS.keys())
        selected = np.random.choice(scenarios, size=num_scenarios, replace=False)
        
        # Générer des poids aléatoires (somme = 1)
        weights = np.random.dirichlet(np.ones(num_scenarios))
        scenario_weights = {s: w for s, w in zip(selected, weights)}
        
        wind_init, wind_evol = sample_scenario_mix(scenario_weights)
        
        weight_str = ", ".join([f"{s}:{w:.2f}" for s, w in scenario_weights.items()])
        return (wind_init, wind_evol, f"Mix: {weight_str}")
    
    def _sample_random_all(self):
        """Phase 6: Tout le spectre."""
        # 40% scénario pur
        # 30% interpolation paire
        # 30% mix pondéré
        rand = np.random.random()
        
        if rand < 0.4:
            return self._sample_balanced_scenarios()
        elif rand < 0.7:
            return self._sample_pairwise_interpolation()
        else:
            return self._sample_weighted_mix()


def generate_interpolated_curriculum_params(episode: int, 
                                             total_episodes: int = 80000) -> Tuple[Dict, Dict, str]:
    """
    Fonction compatible avec votre code existant.
    
    Usage dans my_agent_DQN.py:
        from scenario_interpolation_curriculum import generate_interpolated_curriculum_params
        
        # Dans train():
        init_params, evol_params, description = \
            generate_interpolated_curriculum_params(episode, num_episodes)
    
    Returns:
        wind_init_params, wind_evol_params, description
    """
    curriculum = ScenarioInterpolationCurriculum(total_episodes)
    return curriculum.sample_params(episode)


# Pour testing
if __name__ == "__main__":
    curriculum = ScenarioInterpolationCurriculum(80000)
    
    print("=" * 80)
    print("SCENARIO INTERPOLATION CURRICULUM - Test")
    print("=" * 80)
    
    # Tester quelques épisodes de chaque phase
    test_episodes = [0, 5000, 10000, 20000, 35000, 55000, 70000, 78000]
    
    for ep in test_episodes:
        stage = curriculum.get_current_stage(ep)
        init, evol, desc = curriculum.sample_params(ep)
        
        print(f"\nEpisode {ep} - Stage: {stage['name']}")
        print(f"  Strategy: {stage['strategy']}")
        print(f"  Description: {desc}")
        print(f"  Wind direction: {init['base_direction']}")
        print(f"  Wind change prob: {evol['wind_change_prob']:.2f}")
        print(f"  Pattern strength: {init['pattern_strength']:.2f}")