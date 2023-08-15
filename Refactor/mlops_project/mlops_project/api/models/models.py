from pydantic import BaseModel


class OnlineTX(BaseModel):
    """
    Represents an Online transaction and has various attributes.

    Attributes:
        type (float): Placeholder for type of online transaction.
        amount (float): Placeholder for the amount of the transaction.
        oldbalanceOrg (float): Placeholder for balance before the transaction.
        newbalanceOrg (float): Placeholder for balance after the transaction.
    """

    type: float
    amount: float
    oldbalanceOrg: float
    newbalanceOrg: float
