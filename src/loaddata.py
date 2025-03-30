import redis
from datetime import datetime

# Function to add a guest user to Redis
def add_guest_user(redis_client, euid, name, email, created_by, updated_by):
    user_key = f"user:{euid}"
    user_data = {
        'name': name,
        'EUID': euid,
        'email': email,
        'role': 'guest',
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'created_by': created_by,
        'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'updated_by': updated_by
    }
    redis_client.hset(user_key, mapping=user_data)
    print(f"Guest user {name} added with EUID {euid}")

# Main function to set up Redis connection and add a guest user
def main():
    # Connect to Redis
    try:
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            password='Genai@123',  # Replace with your Redis password if set
            decode_responses=True
        )
        # Test the connection
        redis_client.ping()
        print("Connected to Redis successfully.")
    except redis.ConnectionError as e:
        print(f"Could not connect to Redis: {e}")
        return

    # Example usage: Add a guest user
    add_guest_user(
        redis_client,
        euid='guest123',
        name='John Doe',
        email='john.doe@example.com',
        created_by='admin',
        updated_by='admin'
    )

if __name__ == "__main__":
    main()
