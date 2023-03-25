import matplotlib.pyplot as plt

def plot_images(titles, images):
    rows = 13
    columns = 3

    fig = plt.figure(figsize=(columns*4,rows*4))
    col_index = 1

    for i in range(3):
        image = images[i]
        title = titles[i]
        
        fig.add_subplot(rows, columns, col_index)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')
        col_index += 1

    for i in range(12):
        image = images[i+3]
        title = titles[i+3]
        
        fig.add_subplot(rows, columns, col_index)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')
        
        image = images[i+15]
        title = titles[i+15]
        
        fig.add_subplot(rows, columns, col_index+1)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')
        
        image = images[i+27]
        title = titles[i+27]
        
        fig.add_subplot(rows, columns, col_index+2)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')
        
        col_index += 3

    plt.show()

def plot_images_group(titles, images, rows, columns):
    fig = plt.figure(figsize=(columns*4,rows*4))
    col_index = 1

    for i in range(len(titles)):
        image = images[i]
        title = titles[i]
        
        fig.add_subplot(rows, columns, col_index)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize = 8)
        plt.axis('off')

        col_index += 1

    plt.show()