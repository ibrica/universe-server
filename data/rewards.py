"""
Shared rewards data module
"""
rewards_list = []

def save_reward(averageReward, minReward, maxReward):
    """Add new reward in the list"""
    rewards_list.append({'averageReward':averageReward, 'minReward':minReward, 'maxReward':maxReward})

def get_rewards():
    """Get all rewards"""
    return rewards_list

def reset_rewards():
    """Get all rewards"""
    global rewards_list
    rewards_list = []
    