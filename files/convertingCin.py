import os
import random
import string


def generate_random_filename():
    # Générer un nom de fichier aléatoire avec 6 chiffres et 2 lettres
    random_digits = ''.join(random.choices(string.digits, k=6))
    random_letters = ''.join(random.choices(string.ascii_lowercase, k=2))
    random_filename = f"{random_digits}_{random_letters}"
    return random_filename


def rename_images_with_random_names(input_folder):
    # Vérifier si le chemin donné est un dossier
    if not os.path.isdir(input_folder):
        print("Le chemin spécifié n'est pas un dossier.")
        return

    # Parcourir les fichiers dans le dossier d'entrée
    for filename in os.listdir(input_folder):
        # Construire le chemin absolu du fichier
        filepath = os.path.join(input_folder, filename)
        # Vérifier si c'est un fichier
        if os.path.isfile(filepath):
            # Générer un nouveau nom de fichier aléatoire
            new_filename = generate_random_filename()
            # Construire le nouveau chemin absolu avec le nouveau nom de fichier
            new_filepath = os.path.join(input_folder, new_filename)
            # Renommer le fichier avec le nouveau nom de fichier
            os.rename(filepath, new_filepath)


# Exemple d'utilisation de la fonction
input_folder_path = r'C:\Users\yassi\PycharmProjects\PfeProject\verso'
rename_images_with_random_names(input_folder_path)

# import os
# import shutil
#
#
# def extract_and_rename_images(input_folder, output_folder):
#     # Créer le dossier de sortie s'il n'existe pas
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Parcourir les dossiers dans le dossier d'entrée
#     for subdir in os.listdir(input_folder):
#         subdir_path = os.path.join(input_folder, subdir)
#         # Vérifier si c'est un dossier
#         if os.path.isdir(subdir_path):
#             # Parcourir récursivement les sous-dossiers
#             extract_and_rename_images_recursive(subdir_path, output_folder)
#
#
# def extract_and_rename_images_recursive(input_folder, output_folder):
#     # Parcourir récursivement les dossiers et fichiers dans le dossier d'entrée
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             # Vérifier si le fichier est une image
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                 # Construire le chemin absolu du fichier d'entrée
#                 input_file_path = os.path.join(root, file)
#                 # Construire le nouveau nom de fichier en utilisant le chemin absolu
#                 relative_path = os.path.relpath(input_file_path, input_folder)
#                 new_file_name = relative_path.replace(os.path.sep, '_')
#                 # Construire le chemin absolu du fichier de sortie
#                 output_file_path = os.path.join(output_folder, new_file_name)
#                 # Copier le fichier vers le dossier de sortie et le renommer
#                 shutil.copyfile(input_file_path, output_file_path)
#
#
#
#
# # Exemple d'utilisation de la fonction
# input_folder_path = r'C:\Users\yassi\PycharmProjects\PfeProject\cin'
# output_folder_path = r'C:\Users\yassi\PycharmProjects\PfeProject\outputCin'
# extract_and_rename_images(input_folder_path, output_folder_path)
