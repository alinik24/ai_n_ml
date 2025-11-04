# Literature References

Complete bibliography for the Autonomous Mobile Robot Navigation project. Use these citations in your report.

---

## 📚 Core Papers (Primary Citations)

### 1. Deep Q-Network (DQN)

**Full Citation:**
> Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

**BibTeX:**
```bibtex
@article{mnih2015human,
  title={Human-level control through deep reinforcement learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A and Veness, Joel and Bellemare, Marc G and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K and Ostrovski, Georg and others},
  journal={Nature},
  volume={518},
  number={7540},
  pages={529--533},
  year={2015},
  publisher={Nature Publishing Group}
}
```

**Key Contributions:**
- First deep RL algorithm to achieve human-level performance on Atari games
- Introduced experience replay for stable training
- Combined deep neural networks with Q-learning
- Demonstrated end-to-end learning from raw pixels

**Impact:** 12,000+ citations | Nature (Impact Factor: 64.8)

---

### 2. Double DQN

**Full Citation:**
> van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 30, No. 1).

**BibTeX:**
```bibtex
@inproceedings{van2016deep,
  title={Deep reinforcement learning with double Q-learning},
  author={Van Hasselt, Hado and Guez, Arthur and Silver, David},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={30},
  number={1},
  year={2016}
}
```

**Key Contributions:**
- Addresses Q-value overestimation in standard DQN
- Decouples action selection from action evaluation
- Uses online network for selection, target network for evaluation
- Reduces overestimation bias by 30-50%

**Mathematical Formulation:**
```
Standard DQN: Q_target = r + γ * max_a' Q_target(s', a')
Double DQN:   Q_target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
```

**Impact:** 2,500+ citations | AAAI 2016 Best Paper Nominee

---

### 3. Dueling DQN

**Full Citation:**
> Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. In *International Conference on Machine Learning* (pp. 1995-2003). PMLR.

**BibTeX:**
```bibtex
@inproceedings{wang2016dueling,
  title={Dueling network architectures for deep reinforcement learning},
  author={Wang, Ziyu and Schaul, Tom and Hessel, Matteo and Hasselt, Hado and Lanctot, Marc and Freitas, Nando},
  booktitle={International Conference on Machine Learning},
  pages={1995--2003},
  year={2016},
  organization={PMLR}
}
```

**Key Contributions:**
- Separates Q-function into value V(s) and advantage A(s,a) streams
- Better generalization across actions with similar values
- Particularly effective for navigation and control tasks
- Can be combined with Double DQN

**Mathematical Formulation:**
```
Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
```

**Impact:** 1,800+ citations | ICML 2016 Best Paper Award

---

### 4. Q-Learning (Original)

**Full Citation:**
> Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

**BibTeX:**
```bibtex
@article{watkins1992q,
  title={Q-learning},
  author={Watkins, Christopher JCH and Dayan, Peter},
  journal={Machine learning},
  volume={8},
  number={3-4},
  pages={279--292},
  year={1992},
  publisher={Springer}
}
```

**Key Contributions:**
- Foundational off-policy temporal difference learning algorithm
- Convergence guarantees under certain conditions
- Model-free reinforcement learning
- Bellman optimality equation for Q-values

**Impact:** 18,000+ citations | Foundation of modern RL

---

## 📖 Supporting References

### 5. Experience Replay

**Full Citation:**
> Lin, L. J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning*, 8(3-4), 293-321.

**BibTeX:**
```bibtex
@article{lin1992self,
  title={Self-improving reactive agents based on reinforcement learning, planning and teaching},
  author={Lin, Long-Ji},
  journal={Machine learning},
  volume={8},
  number={3-4},
  pages={293--321},
  year={1992},
  publisher={Springer}
}
```

**Key Contribution:** Introduced experience replay buffer for breaking temporal correlations

---

### 6. Reinforcement Learning Textbook

**Full Citation:**
> Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

**BibTeX:**
```bibtex
@book{sutton2018reinforcement,
  title={Reinforcement learning: An introduction},
  author={Sutton, Richard S and Barto, Andrew G},
  year={2018},
  publisher={MIT press},
  edition={2nd}
}
```

**Key Topics:** Epsilon-greedy exploration, temporal difference learning, Bellman equations

---

### 7. Rainbow DQN (Extension)

**Full Citation:**
> Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., ... & Silver, D. (2018). Rainbow: Combining improvements in deep reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 32, No. 1).

**BibTeX:**
```bibtex
@inproceedings{hessel2018rainbow,
  title={Rainbow: Combining improvements in deep reinforcement learning},
  author={Hessel, Matteo and Modayil, Joseph and Van Hasselt, Hado and Schaul, Tom and Ostrovski, Georg and Dabney, Will and Horgan, Dan and Piot, Bilal and Azar, Mohammad and Silver, David},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={32},
  number={1},
  year={2018}
}
```

**Note:** Rainbow combines Double DQN, Dueling DQN, and other improvements. Reference for future work.

---

### 8. Gymnasium (OpenAI Gym Successor)

**Full Citation:**
> Towers, M., Terry, J. K., Kwiatkowski, A., Balis, J. U., Cola, G. D., Deleu, T., ... & Younis, O. G. (2023). Gymnasium. *Zenodo*.

**BibTeX:**
```bibtex
@software{gymnasium2023,
  title={Gymnasium},
  author={Towers, Mark and Terry, Jordan K and Kwiatkowski, Ariel and Balis, John U and Cola, Gianluca De and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and KG, Arjun and Krimmel, Markus and others},
  year={2023},
  publisher={Zenodo},
  url={https://github.com/Farama-Foundation/Gymnasium}
}
```

---

### 9. PyTorch

**Full Citation:**
> Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems* (pp. 8024-8035).

**BibTeX:**
```bibtex
@inproceedings{paszke2019pytorch,
  title={PyTorch: An imperative style, high-performance deep learning library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory andKilleen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle={Advances in neural information processing systems},
  pages={8024--8035},
  year={2019}
}
```

---

## 🎯 Related Work (Optional Context)

### Autonomous Mobile Robots

**Full Citation:**
> Koenig, S., & Simmons, R. G. (1998). Xavier: A robot navigation architecture based on partially observable Markov decision process models. In *Artificial Intelligence Based Mobile Robotics* (pp. 91-122).

**BibTeX:**
```bibtex
@incollection{koenig1998xavier,
  title={Xavier: A robot navigation architecture based on partially observable Markov decision process models},
  author={Koenig, Sven and Simmons, Reid G},
  booktitle={Artificial Intelligence Based Mobile Robotics},
  pages={91--122},
  year={1998},
  publisher={MIT Press}
}
```

---

### Deep RL for Robotics

**Full Citation:**
> Levine, S., Finn, C., Darrell, T., & Abbeel, P. (2016). End-to-end training of deep visuomotor policies. *Journal of Machine Learning Research*, 17(39), 1-40.

**BibTeX:**
```bibtex
@article{levine2016end,
  title={End-to-end training of deep visuomotor policies},
  author={Levine, Sergey and Finn, Chelsea and Darrell, Trevor and Abbeel, Pieter},
  journal={Journal of Machine Learning Research},
  volume={17},
  number={39},
  pages={1--40},
  year={2016}
}
```

---

## 📝 How to Use These References in Your Report

### In-Text Citation Examples

**IEEE Style (Recommended for ML/Engineering):**
```
Standard DQN [1] uses a single target network, while Double DQN [2]
decouples action selection and evaluation to reduce overestimation bias.
```

**APA Style:**
```
Mnih et al. (2015) demonstrated human-level performance on Atari games
using deep Q-networks. Later, van Hasselt et al. (2016) addressed
overestimation through Double Q-learning.
```

### Section-Specific Citations

**Introduction:**
- Cite foundational papers: Watkins & Dayan (1992), Sutton & Barto (2018)
- Cite DRL breakthrough: Mnih et al. (2015)

**Related Work:**
- Cite all core papers: Mnih (2015), van Hasselt (2016), Wang (2016)
- Cite robotics applications: Koenig & Simmons (1998), Levine et al. (2016)

**Methods:**
- Cite specific algorithms you implement
- Q-Learning: Watkins & Dayan (1992)
- DQN: Mnih et al. (2015)
- Double DQN: van Hasselt et al. (2016)
- Dueling DQN: Wang et al. (2016)

**Implementation:**
- Cite frameworks: Towers et al. (2023) for Gymnasium, Paszke et al. (2019) for PyTorch

**Results:**
- Compare to benchmarks from cited papers
- Reference expected performance ranges

**Discussion:**
- Relate your findings to published results
- Discuss improvements from Double/Dueling DQN papers

---

## 📊 Citation Summary

| Paper | Year | Venue | Citations | Relevance |
|-------|------|-------|-----------|-----------|
| Q-Learning | 1992 | Machine Learning | 18,000+ | Foundation |
| Experience Replay | 1992 | Machine Learning | 10,000+ | Core technique |
| RL Textbook | 2018 | MIT Press | 25,000+ | Theory |
| DQN | 2015 | Nature | 12,000+ | **Primary** |
| Double DQN | 2016 | AAAI | 2,500+ | **Primary** |
| Dueling DQN | 2016 | ICML | 1,800+ | **Primary** |
| Rainbow DQN | 2018 | AAAI | 1,200+ | Extension |
| Gymnasium | 2023 | Software | N/A | Framework |
| PyTorch | 2019 | NeurIPS | 8,000+ | Framework |

---

## 🎓 Academic Integrity Note

**IMPORTANT:** Always cite sources properly to:
- Give credit to original authors
- Allow readers to verify claims
- Avoid plagiarism
- Demonstrate scholarly rigor

**Citation Guidelines:**
1. Cite whenever you describe someone else's work or ideas
2. Use direct quotes sparingly and always with quotation marks
3. Paraphrase in your own words when possible
4. Include both in-text citations and full bibliography
5. Use consistent citation style throughout

---

## 📚 Additional Resources

### Online Resources
- **DeepMind Research:** https://www.deepmind.com/research
- **OpenAI Spinning Up:** https://spinningup.openai.com/
- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **Gymnasium Docs:** https://gymnasium.farama.org/

### Review Papers
- Arulkumaran et al. (2017). "Deep Reinforcement Learning: A Brief Survey." *IEEE Signal Processing Magazine*.
- Li, Y. (2017). "Deep reinforcement learning: An overview." *arXiv preprint arXiv:1701.07274*.

---

**Last Updated:** 2025-11-03

**Usage:** Copy relevant BibTeX entries to your report's bibliography section.
