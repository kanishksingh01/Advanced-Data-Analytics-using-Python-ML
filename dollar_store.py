def calculate_total_cost(item_count):
    item_price = 1  # Price of each item
    discount_threshold = 10  # Number of items required for discount
    discount_percentage = 0.05  # 5% discount
    sales_tax_percentage = 0.075  # 7.5% sales tax

    total_cost = item_count * item_price  # Calculate total cost without discount

    if item_count > discount_threshold:
        discount_amount = total_cost * discount_percentage
        total_cost -= discount_amount

    sales_tax_amount = total_cost * sales_tax_percentage
    total_cost += sales_tax_amount

    return total_cost

item_count = int(input("Enter the number of items: "))
total_cost = calculate_total_cost(item_count)

gross_cost = item_count * 1
discount = 0
net_cost = gross_cost

if item_count > 10:
    discount = gross_cost * 0.05
    net_cost = gross_cost - discount

tax = net_cost * 0.075
total_price = net_cost + tax

print(f"Gross cost: ${gross_cost:.2f}")
print(f"Discount: ${discount:.2f}")
print(f"Net cost: ${net_cost:.2f}")
print("")
print(f"Tax: ${tax:.2f}")
print(f"Total price after tax: ${total_price:.2f}")

